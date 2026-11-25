#include "common/assert.h"
#include "common/liveControl.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "gpu_blit_shaders/gpu_pssr_linear_spv.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/upscaler/frameGeneration.h"
#include "graphics/host_gpu/renderer/upscaler/fsr.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <optional>

namespace Libs::Graphics {
namespace {

struct PssrParameters {
	uint32_t input_size[2];
	uint32_t output_size[2];
	uint32_t output_channels[4];
};

// Upscalers read and write with stages and accesses of their own.
void MemoryBarrier(vk::CommandBuffer command) {
	const vk::MemoryBarrier2 barrier {.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
	                                  .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
	                                                   vk::AccessFlagBits2::eMemoryWrite};
	command.pipelineBarrier2(
	    vk::DependencyInfo {.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

// 0: the game's own upscaler makes the picture although the emulator's is loaded (a comparison).
Common::LiveControl::Switch g_game_upscaler("game_upscaler", 1);
// 0: the game's last pass runs although the emulator's upscaler makes the picture over it.
Common::LiveControl::Switch g_game_upscaler_skip("game_upscaler_skip", 1);

} // namespace

RenderExecutor::~RenderExecutor() {
	const auto device = m_context.GetGraphics().device;
	device.destroyPipeline(m_pssr_pipeline, nullptr);
	device.destroyPipelineLayout(m_pssr_pipeline_layout, nullptr);
	device.destroyDescriptorSetLayout(m_pssr_descriptor_layout, nullptr);
}

bool RenderExecutor::Upscale(vk::CommandBuffer command, const Pssr::Request& request,
                             Dlss::Inputs& inputs, bool any_size) {
	const auto release = [&](std::unique_ptr<Upscaler::Feature>& feature) {
		if (feature != nullptr) {
			m_context.GetCommandScheduler().DeferOperation(
			    [feature = std::move(feature)]() mutable { feature.reset(); });
		}
	};
	auto context =
	    std::ranges::find(m_upscaler_contexts, request.context, &UpscalerContext::context);
	bool reset = (request.flags & Pssr::kReset) != 0;
	if (context == m_upscaler_contexts.end() || !Dlss::Takes(context->creation, request, inputs)) {
		// A render size the upscaler can't take scales linearly.
		const auto creation = Upscaler::Choose(request, inputs, any_size);
		if (!creation) {
			return false;
		}
		if (context != m_upscaler_contexts.end()) {
			release(context->feature);
		} else {
			// A library has at most four contexts.
			if (m_upscaler_contexts.size() == 4) {
				release(m_upscaler_contexts.front().feature);
				m_upscaler_contexts.erase(m_upscaler_contexts.begin());
			}
			context = m_upscaler_contexts.insert(m_upscaler_contexts.end(),
			                                     {request.context, {}, nullptr});
		}
		context->creation = *creation;
		context->feature  = Upscaler::Create(command, *creation);
		reset             = true;
	}
	if (context->feature == nullptr) {
		return false;
	}
	MemoryBarrier(command);
	if (!context->feature->Evaluate(command, request, inputs, reset)) {
		release(context->feature);
		return false;
	}
	MemoryBarrier(command);
	Dlss::NoteEvaluation(m_context.GetCommandScheduler().CurrentTick());
	return true;
}

void RenderExecutor::ScaleLinear(vk::CommandBuffer command, vk::ImageView source,
                                 vk::Extent2D source_size, vk::ImageView destination,
                                 vk::Extent2D destination_size,
                                 const std::array<uint32_t, 4>& channels) {
	auto& graphics = m_context.GetGraphics();
	if (m_pssr_pipeline == nullptr) {
		const std::array bindings {
		    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eSampledImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute}};
		vk::DescriptorSetLayoutCreateInfo descriptor_info {};
		descriptor_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptor_info.bindingCount = bindings.size();
		descriptor_info.pBindings = bindings.data();
		RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(
		                         &descriptor_info, nullptr, &m_pssr_descriptor_layout),
		                     "create upscaling descriptor layout");
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0,
		                                  sizeof(PssrParameters)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount = 1;
		layout_info.pSetLayouts = &m_pssr_descriptor_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges = &push;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(
		                         &layout_info, nullptr, &m_pssr_pipeline_layout),
		                     "create upscaling pipeline layout");
		m_pssr_pipeline =
		    CreateComputePipeline(graphics.device, GPU_PSSR_LINEAR_SPV, m_pssr_pipeline_layout,
			                      "create upscaling pipeline");
	}
	const PssrParameters parameters {{source_size.width, source_size.height},
	                                 {destination_size.width, destination_size.height},
	                                 {channels[0], channels[1], channels[2], channels[3]}};
	const std::array images {
	    vk::DescriptorImageInfo {nullptr, source, vk::ImageLayout::eGeneral},
	    vk::DescriptorImageInfo {nullptr, destination, vk::ImageLayout::eGeneral}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = i == 0 ? vk::DescriptorType::eSampledImage
		                                 : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo = &images[i];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pssr_pipeline_layout,
	                             0, writes.size(), writes.data());
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pssr_pipeline);
	command.pushConstants(m_pssr_pipeline_layout, vk::ShaderStageFlagBits::eCompute,
	                      0, sizeof(parameters), &parameters);
	command.dispatch((parameters.output_size[0] + 7) / 8,
	                 (parameters.output_size[1] + 7) / 8, 1);
	ShaderWriteBarrier(command, vk::PipelineStageFlagBits::eComputeShader);
}

// Whether the game's last upscaler pass is left out of this frame: the emulator's upscaler took
// the frame before at this render size, so it takes this one, and it makes all the pass makes
// that anything reads (the picture, and the small copy of it by ScaleLinear).
bool RenderExecutor::SkipsGamePass(const FrameGen::UpscalerPass& pass) const {
	return Upscaler::Enabled() && g_game_upscaler.On() && g_game_upscaler_skip.On() &&
	       pass.reduced != FrameGen::UpscalerPass::NoImage && pass.width != 0 &&
	       m_game_upscaler_taken == (uint64_t {pass.width} << 32u | pass.height);
}

// The game has no PSSR (base mode), but its engine's own temporal upscaler takes the same inputs:
// the frame's color, depth and motion vectors at the render size, the sample's jitter, and makes
// the picture at the output size. Its passes run as they are but the last one, which makes the
// picture: that is made here instead, into the layer of the game's history that pass writes, so
// the game's next frame and everything after the upscaler read it as the game's own. The first
// frame of a render size runs the game's pass too (`skipped` false: whether the upscaler takes
// the size is known only once it did); a frame the upscaler does not take keeps the game's
// picture. What the last pass writes besides: a small copy of the picture, made here from the
// picture, and its history's weights, which only that pass reads.
void RenderExecutor::UpscaleGameFrame(CommandBuffer& buffer, const PreparedBindings& bindings,
                                      const FrameGen::UpscalerPass& pass, uint64_t shader_hash,
                                      bool skipped) {
	const CommandBuffer::ProfileGuard profile(buffer, CommandBuffer::ProfileScope::Upscaler);
	auto* capture = FrameGen::TakeCaptured();
	if (capture == nullptr || !Upscaler::Enabled() || !g_game_upscaler.On()) {
		m_game_upscaler_taken = 0;
		return;
	}
	constexpr uint64_t CONTEXT = 0x52535455'454d4147; // "GAMEUTSR": the one history there is
	auto&              cache   = m_context.GetTextureCache();
	auto&              color   = cache.GetImage(bindings.images[pass.color].image_id);
	auto&              output  = cache.GetImage(bindings.images[pass.output].image_id);
	auto&              images  = *capture->images;
	const auto&        view    = capture->view;
	// One level and one layer of each, as a 2D image: the layer of the history the pass writes.
	const auto single = [](ImageViewInfo info, vk::ImageUsageFlags usage) {
		info.type        = vk::ImageViewType::e2D;
		info.level_count = 1;
		info.layer_count = 1;
		info.min_lod     = 0;
		info.usage       = usage;
		return info;
	};
	const auto color_info  = single(bindings.images[pass.color].desc.view_info,
	                                vk::ImageUsageFlagBits::eSampled);
	const auto output_info = single(bindings.images[pass.output].desc.view_info,
	                                vk::ImageUsageFlagBits::eStorage);
	const auto color_extent  = color.info.MipExtent(color_info.base_level);
	const auto output_extent = output.info.MipExtent(output_info.base_level);
	if (!color.registered || !output.registered || color.backing.image == nullptr ||
	    output.backing.image == nullptr || color.backing.samples != 1 ||
	    output.backing.samples != 1 || color.info.IsDepth() || output.info.IsDepth() ||
	    view.width > color_extent.width || view.height > color_extent.height ||
	    view.width > images.depth.extent.width || view.height > images.depth.extent.height ||
	    output_extent.width < view.width || output_extent.height < view.height) {
		m_game_upscaler_taken = 0;
		return;
	}
	const auto resource = [](Image& image, const ImageViewInfo& info, vk::Extent3D extent,
	                         bool write) {
		return Dlss::Resource {
		    .view       = static_cast<VkImageView>(image.FindView(info)),
		    .image      = static_cast<VkImage>(image.backing.image),
		    .range      = {static_cast<VkImageAspectFlags>(info.aspect), info.base_level, 1,
		                   info.base_layer, 1},
		    .format     = static_cast<VkFormat>(info.format),
		    .width      = extent.width,
		    .height     = extent.height,
		    .read_write = write};
	};
	const auto captured = [](const VulkanImage& image, vk::ImageView image_view,
	                         vk::ImageAspectFlags aspect) {
		return Dlss::Resource {.view   = static_cast<VkImageView>(image_view),
		                       .image  = static_cast<VkImage>(image.image),
		                       .range  = {static_cast<VkImageAspectFlags>(aspect), 0, 1, 0, 1},
		                       .format = static_cast<VkFormat>(image.format),
		                       .width  = image.extent.width,
		                       .height = image.extent.height};
	};
	Pssr::Request request {};
	request.context = CONTEXT;
	request.width   = view.width;
	request.height  = view.height;
	// (The engine's offset is the upscaler's as it is: negated in x the picture blurs, in y its
	// edges come out as stairs.)
	request.jitter[0] = view.jitter[0];
	request.jitter[1] = view.jitter[1];
	// (The captured vectors are in parts of the frame.)
	request.motion_scale[0] = view.motion_scale[0] * static_cast<float>(view.width);
	request.motion_scale[1] = view.motion_scale[1] * static_cast<float>(view.height);
	request.pre_exposure    = view.pre_exposure;
	request.flags = Pssr::kReverseDepth | Pssr::kAutoExposure | (view.cut ? Pssr::kReset : 0u);

	buffer.EndRendering();
	// (The upscaler binds its own pipeline: the next guest dispatch or draw binds its own again.)
	const auto command = buffer.PipelineHandle();
	color.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
	              ImageSubresourceRange {color_info.base_level, 1, color_info.base_layer, 1},
	              command);
	output.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
	               ImageSubresourceRange {output_info.base_level, 1, output_info.base_layer, 1},
	               command);
	FrameGen::SetUpscalerLayout(command, images, true);
	Dlss::Inputs inputs {resource(color, color_info, color_extent, false),
	                     captured(images.depth, images.depth_view,
	                              vk::ImageAspectFlagBits::eDepth),
	                     captured(images.motion, images.motion_view,
	                              vk::ImageAspectFlagBits::eColor),
	                     nullptr, resource(output, output_info, output_extent, true)};
	// (FidelityFX views an image from its first layer: not this one.)
	const bool upscaled = !Fsr::Enabled() && Upscale(command, request, inputs, true);
	FrameGen::SetUpscalerLayout(command, images, false);
	m_game_upscaler_taken = upscaled ? uint64_t {view.width} << 32u | view.height : 0;
	if (upscaled && skipped) {
		// The small copy the game's pass would have written.
		auto&      reduced        = cache.GetImage(bindings.images[pass.reduced].image_id);
		const auto reduced_info   = single(bindings.images[pass.reduced].desc.view_info,
		                                   vk::ImageUsageFlagBits::eStorage);
		const auto reduced_extent = reduced.info.MipExtent(reduced_info.base_level);
		if (reduced.registered && reduced.backing.image != nullptr &&
		    reduced.backing.samples == 1) {
			reduced.Transit(
			    vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
			    ImageSubresourceRange {reduced_info.base_level, 1, reduced_info.base_layer, 1},
			    command);
			ScaleLinear(command,
			            output.FindView(single(bindings.images[pass.output].desc.view_info,
			                                   vk::ImageUsageFlagBits::eSampled)),
			            {output_extent.width, output_extent.height}, reduced.FindView(reduced_info),
			            {reduced_extent.width, reduced_extent.height}, {0, 1, 2, 4});
		}
	}
	// (Once per shader, sizes and outcome: what was done.)
	const uint64_t state = shader_hash ^ (uint64_t {view.width} << 48u) ^
	                       (uint64_t {view.height} << 32u) ^ (uint64_t {output_extent.width} << 16u) ^
	                       (uint64_t {skipped} << 1u) ^ uint64_t {upscaled};
	if (m_game_upscaler_logged != state) {
		m_game_upscaler_logged = state;
		LOGF("game upscaler: the picture of CS 0x%016" PRIx64 " %s: %ux%u of %ux%u to %ux%u "
		     "(layer %u of %u, pre-exposure %g)\n",
		     shader_hash,
		     !upscaled ? (skipped ? "NOT MADE (the upscaler refused a size it took before)"
		                          : "left as the game made it")
		               : (skipped ? "made in the game's place" : "made again over the game's"),
		     view.width, view.height, color_extent.width, color_extent.height, output_extent.width,
		     output_extent.height, output_info.base_layer, output.backing.layers,
		     static_cast<double>(view.pre_exposure));
	}
}

void RenderExecutor::ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request) {
	m_context.GetCommandScheduler().TryPopPendingOperations();
	Common::LockGuard lock(m_context.GetMutex());
	ShaderTextureResource input_descriptor {}, output_descriptor {};
	std::copy_n(request.input, 8, input_descriptor.fields);
	std::copy_n(request.output, 8, output_descriptor.fields);
	const auto is_2d = [](const ShaderTextureResource& descriptor) {
		return !descriptor.IsNull() &&
		       (descriptor.Type() == Prospero::ImageType::kColor2D ||
		        descriptor.Type() == Prospero::ImageType::kColor2DArray) &&
		       descriptor.TileMode() != Prospero::TileMode::kDepth;
	};
	EXIT_IF(!is_2d(input_descriptor) || !is_2d(output_descriptor) ||
	        !Pssr::IsColorFormat(input_descriptor.Format()) ||
	        output_descriptor.Format() != Prospero::BufferFormat::k11_11_10Float ||
	        output_descriptor.TileMode() != Prospero::TileMode::kStandard256B ||
	        output_descriptor.BaseLevel() != 0 || output_descriptor.LastLevel() != 0 ||
	        output_descriptor.MaxMip() != 0 || output_descriptor.BaseArray5() != 0 ||
	        output_descriptor.Depth() != 0 || request.width == 0 || request.height == 0);
	const auto input_width = std::max(1u, (input_descriptor.Width5() + 1u) >>
	                                         input_descriptor.BaseLevel());
	const auto input_height = std::max(1u, (input_descriptor.Height5() + 1u) >>
	                                          input_descriptor.BaseLevel());
	EXIT_IF(request.width > input_width || request.height > input_height);
	PssrParameters parameters {{request.width, request.height},
	                           {output_descriptor.Width5() + 1u, output_descriptor.Height5() + 1u},
	                           {4, 4, 4, 4}};
	for (uint32_t component = 0; component < 3; ++component) {
		for (uint32_t source = 0; source < 4; ++source) {
			if (GetDstSel(output_descriptor.DstSelXYZW(), source) == 4u + component) {
				parameters.output_channels[component] = source;
				break;
			}
		}
	}
	const auto resolve = [&](const uint32_t* words, bool output) {
		ShaderRecompiler::IR::ImageResource resource {};
		resource.resource_class = output ? ShaderRecompiler::IR::ImageResourceClass::Storage
		                                 : ShaderRecompiler::IR::ImageResourceClass::Sampled;
		resource.numeric_class = Prospero::TextureNumericClass::Float;
		resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		resource.read = !output;
		resource.written = output;
		ShaderRecompiler::IR::DescriptorValue value {};
		std::copy_n(words, 8, value.dwords.begin());
		value.dword_count = 8;
		auto binding = ResolveTexture(resource, value);
		binding.desc.view_info.level_count = 1;
		binding.desc.view_info.min_lod = 0;
		return binding;
	};
	auto& cache = m_context.GetTextureCache();
	auto input = resolve(request.input, false);
	BindImage(input.image_id, false);
	auto output = resolve(request.output, true);
	BindImage(output.image_id, true);
	EXIT_IF(input.desc.info.data.address < output.desc.info.data.End() &&
	        output.desc.info.data.address < input.desc.info.data.End());
	// Depth, motion vectors and exposure for DLSS or FSR, which write the output unswizzled.
	const std::array<const uint32_t*, 3>         dlss_words {request.depth, request.motion,
	                                                         request.exposure};
	std::array<std::optional<TextureBinding>, 3> dlss_bindings;
	if (Upscaler::Enabled() && Dlss::Accepts(request) && parameters.output_channels[0] == 0 &&
	    parameters.output_channels[1] == 1 && parameters.output_channels[2] == 2) {
		for (size_t i = 0; i < dlss_words.size(); ++i) {
			ShaderTextureResource descriptor {};
			std::copy_n(dlss_words[i], 8, descriptor.fields);
			if (!descriptor.IsNull()) {
				dlss_bindings[i] = resolve(dlss_words[i], false);
				BindImage(dlss_bindings[i]->image_id, false);
			}
		}
	}
	const auto acquire = [&](TextureBinding& binding, const uint32_t* words, bool storage) {
		auto& previous = cache.GetImage(binding.image_id);
		if (!previous.registered || previous.binding.needs_rebind) {
			previous.binding = {};
			binding = resolve(words, storage);
			BindImage(binding.image_id, storage);
		}
		binding.image_view = cache.FindTexture(binding.image_id, binding.desc);
	};
	for (size_t i = 0; i < dlss_words.size(); ++i) {
		if (dlss_bindings[i]) {
			acquire(*dlss_bindings[i], dlss_words[i], false);
		}
	}
	acquire(input, request.input, false);
	acquire(output, request.output, true);
	auto& source = cache.GetImage(input.image_id);
	auto& destination = cache.GetImage(output.image_id);
	const auto& source_view = input.desc.view_info;
	const auto& destination_view = output.desc.view_info;
	EXIT_IF(source.info.IsDepth() || destination.info.IsDepth() ||
	        source.backing.samples != 1 || destination.backing.samples != 1 ||
	        (input.image_id == output.image_id &&
	         source_view.base_level == destination_view.base_level &&
	         source_view.base_layer == destination_view.base_layer));

	buffer.EndRendering();
	// Binds its own pipeline: the next guest dispatch or draw must bind its pipeline again.
	const auto command = buffer.PipelineHandle();
	const auto range   = [](const ImageViewInfo& view) {
		return ImageSubresourceRange {view.base_level, 1, view.base_layer, 1};
	};
	source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, range(source_view),
	               command);
	destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
	                    range(destination_view), command);
	if (dlss_bindings[0]) {
		// Whether a binding is a whole one-level, one-layer image, unswizzled in its channels.
		const auto raw = [&](const TextureBinding& binding, uint32_t channels) {
			const auto&      image  = cache.GetImage(binding.image_id);
			const auto&      view   = binding.desc.view_info;
			const auto       extent = binding.desc.info.MipExtent(view.base_level);
			const std::array swizzles {view.mapping.r, view.mapping.g, view.mapping.b};
			for (uint32_t i = 0; i < channels; ++i) {
				if (swizzles[i] != vk::ComponentSwizzle::eIdentity &&
				    swizzles[i] != static_cast<vk::ComponentSwizzle>(VK_COMPONENT_SWIZZLE_R + i)) {
					return false;
				}
			}
			return view.base_level == 0 && view.base_layer == 0 && image.backing.layers == 1 &&
			       image.backing.mip_levels == 1 && image.backing.samples == 1 &&
			       image.backing.extent.width == extent.width &&
			       image.backing.extent.height == extent.height;
		};
		const auto resource = [&](const TextureBinding& binding, bool write) {
			const auto& image  = cache.GetImage(binding.image_id);
			const auto& view   = binding.desc.view_info;
			const auto  extent = binding.desc.info.MipExtent(view.base_level);
			// Upscalers take a depth image in its backing format and depth aspect.
			const bool depth  = image.info.IsDepth();
			const auto aspect = depth ? vk::ImageAspectFlagBits::eDepth : view.aspect;
			return Dlss::Resource {
			    .view       = static_cast<VkImageView>(binding.image_view),
			    .image      = static_cast<VkImage>(image.backing.image),
			    .range      = {static_cast<VkImageAspectFlags>(aspect), view.base_level, 1,
			                   view.base_layer, 1},
			    .format     = static_cast<VkFormat>(depth ? image.backing.format : view.format),
			    .width      = extent.width,
			    .height     = extent.height,
			    .read_write = write};
		};
		for (const auto& binding: dlss_bindings) {
			if (binding) {
				cache.GetImage(binding->image_id)
				    .Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
					         range(binding->desc.view_info), command);
			}
		}
		auto exposure = dlss_bindings[2] ? resource(*dlss_bindings[2], false) : Dlss::Resource {};
		Dlss::Inputs inputs {resource(input, false), resource(*dlss_bindings[0], false),
		                     resource(*dlss_bindings[1], false),
		                     dlss_bindings[2] ? &exposure : nullptr, resource(output, true)};
		const bool   readable =
		    !Fsr::Enabled() || (Fsr::Reads(request, inputs) && raw(input, 3) &&
		                        raw(*dlss_bindings[0], 1) && raw(*dlss_bindings[1], 2) &&
		                        (!dlss_bindings[2] || raw(*dlss_bindings[2], 1)) && raw(output, 3));
		if (readable && Upscale(command, request, inputs)) {
			ResetBindings();
			return;
		}
	}
	ScaleLinear(command, input.image_view, {parameters.input_size[0], parameters.input_size[1]},
	            output.image_view, {parameters.output_size[0], parameters.output_size[1]},
	            {parameters.output_channels[0], parameters.output_channels[1],
	             parameters.output_channels[2], parameters.output_channels[3]});
	ResetBindings();
}

} // namespace Libs::Graphics
