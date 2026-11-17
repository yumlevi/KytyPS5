#include "graphics/host_gpu/renderer/occlusionQueries.h"

#include "common/liveControl.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>

namespace Libs::Graphics {

namespace {

// Queries no dump has read yet; past them a draw counts one sample.
constexpr uint32_t QUERY_COUNT = 16384;

// 1: a dump's counters are written when the GPU has run it, by the completion thread, as an
// end-of-pipe label is; the GPU thread goes on. 0: the GPU thread waits for the GPU at the next
// label, flush or flip until they are written (Saros: one wait of 2 ms a frame).
Common::LiveControl::Switch g_occlusion_defer("occlusion_defer", 1);

} // namespace

OcclusionQueries::OcclusionQueries(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_scheduler(scheduler) {
	if (!graphics.occlusion_queries_enabled) {
		return;
	}
	m_device = graphics.device;
	const vk::QueryPoolCreateInfo info {.queryType  = vk::QueryType::eOcclusion,
	                                    .queryCount = QUERY_COUNT};
	RequireVulkanSuccess(m_device.createQueryPool(&info, nullptr, &m_pool), "vkCreateQueryPool");
	m_device.resetQueryPool(m_pool, 0, QUERY_COUNT);
	m_results.resize(QUERY_COUNT);
	if (graphics.occlusion_query_precise_enabled) {
		m_control = vk::QueryControlFlagBits::ePrecise;
	}
}

OcclusionQueries::~OcclusionQueries() {
	if (m_pool) {
		m_device.destroyQueryPool(m_pool, nullptr);
	}
}

int32_t OcclusionQueries::BeginQuery(vk::CommandBuffer command) {
	if (m_begun - m_read.load(std::memory_order_acquire) == QUERY_COUNT) {
		m_ledger.Credit();
		return -1;
	}
	const auto query = static_cast<uint32_t>(m_begun++ % QUERY_COUNT);
	command.beginQuery(m_pool, query, m_control);
	return static_cast<int32_t>(query);
}

void OcclusionQueries::Dump(uint64_t address) {
	m_dumps.push_back(m_ledger.Record(address, m_begun));
}

void OcclusionQueries::PublishDumps(bool now) {
	if (!m_dumps.empty()) {
		// The dumps depend only on draws recorded before this call, as a label does: the
		// completion thread runs them, in the order of the queue, once the GPU has. A label
		// queued after this one is written after these counters.
		m_queued_tick = m_scheduler.DeferPriorityOperationAfterRecorded(
		    [this, dumps = std::move(m_dumps)] { Complete(dumps); });
		m_dumps.clear();
	}
	if ((now || !g_occlusion_defer.On()) && m_queued_tick != 0) {
		m_scheduler.Wait(m_queued_tick);
		m_scheduler.WaitPriorityOperations(m_queued_tick);
		m_queued_tick = 0;
		// ... and those the completion thread handed to this thread.
		m_scheduler.Context().WritePendingZPassCounters();
	}
}

void OcclusionQueries::Complete(const std::vector<PendingDump>& dumps) {
	// (Run at once by a scheduler that is shutting down: the GPU may not have run the queries.)
	if (!CommandScheduler::InDeferredOperation()) {
		return;
	}
	const auto last = dumps.back().queries;
	auto       read = m_read.load(std::memory_order_relaxed);
	// The unread queries wrap around the pool at most once.
	for (uint64_t at = read; at < last;) {
		const auto first = static_cast<uint32_t>(at % QUERY_COUNT);
		const auto count = static_cast<uint32_t>(std::min<uint64_t>(last - at, QUERY_COUNT - first));
		RequireVulkanSuccess(m_device.getQueryPoolResults(m_pool, first, count,
		                                                  count * sizeof(uint64_t),
		                                                  m_results.data() + first, sizeof(uint64_t),
		                                                  vk::QueryResultFlagBits::e64),
		                     "vkGetQueryPoolResults");
		m_device.resetQueryPool(m_pool, first, count);
		at += count;
	}
	m_counters.clear();
	for (const auto& dump: dumps) {
		for (; read < dump.queries; read++) {
			m_samples += m_results[read % QUERY_COUNT];
		}
		m_counters.emplace_back(dump.address, m_samples + dump.uncounted);
	}
	// (After the resets: BeginQuery takes a slot again only once it is read.)
	m_read.store(read, std::memory_order_release);
	m_scheduler.Context().DeliverZPassCounters(m_counters);
}

} // namespace Libs::Graphics
