# Changes in this fork

This branch (`saros-port`) starts from the `saros` branch of
[CarRamRod-963/KytyPS5](https://github.com/CarRamRod-963/KytyPS5) (its `saros-v0.1.7` since
2026-11-17, `saros-v0.1.5` before) and adds the work below, most of it for Saros (PPSA07631) on
Windows. Several pieces are ported from, or follow ideas of, the
Demon's Souls fork [chenxiao07/KytyPS5](https://github.com/chenxiao07/KytyPS5); those are marked
"from the DS fork".

Numbers were measured on Saros with an RTX 4090 and an i9-13900K, each as an off/on comparison
inside one running session unless it says otherwise. They belong to the spot and settings named
with them; other places and machines will differ.

## For players

### A launcher for Windows

- `launcher.cmd` opens **KytyLauncher**, a settings window with pages for play, display, graphics,
  game, shaders and advanced options (from the DS fork). It needs no install; build it with
  `launcher-windows\build.cmd`. Without the app, `launcher.cmd` falls back to the script
  `launcher.ps1`, which offers the same settings.
- `run-windows.ps1` starts the emulator from a settings file or from options, and
  `build-windows.cmd` builds it. Both are what the launcher runs underneath.
- Settings are kept in `launcher-settings.json` next to the launcher (not tracked by git).

### Frame generation (DLSS Frame Generation)

- NVIDIA DLSS Frame Generation through NVIDIA Streamline (from the DS fork), with Saros's depth
  and motion vectors: the screen shows two or more frames for every frame the game renders.
- Needs an RTX 40 or newer card and NVIDIA's Streamline SDK, which is not shipped here.
- Launcher row "Frame generation", or `run-windows.ps1 -FrameGen <frames>`.
- Menus, movies and loading screens are shown as they are.
- Details: [docs/frame-generation.md](docs/frame-generation.md).

### DLSS upscaling

- **NVIDIA DLSS makes Saros's 4K picture in place of the game's own upscaler.** The game's
  upscaler breaks thin and distant things up into stairs and dots, the more the smaller the 3D
  it starts from; DLSS makes a clean picture of the same 3D.
- Launcher row "Upscaling", or `run-windows.ps1 -Dlss <quality|balanced|performance|ultra>`:
  the 3D at 2560x1440, 2227x1253, 1920x1080 or 1280x720 (the game's own: 1600x900 to
  2304x1296).
- At one spot (RTX 4090): the game as it is 41.7 fps, DLSS performance 41.3, DLSS ultra 46.7,
  balanced 39.5, quality 32.8. At the same 3D size DLSS costs about what the game's upscaler
  does, because the game's own last pass is left out.
- Needs an NVIDIA RTX card and DLSS's model (`nvngx_dlss.dll`) from NVIDIA's Streamline SDK,
  the folder frame generation uses; both work together. Without them the game's upscaler stays.
- Details: [docs/dlss-upscaling.md](docs/dlss-upscaling.md).

### 3D resolution and the game's own Unreal settings

- Unreal Engine titles keep their console variables; the emulator can now set them in the running
  game without patching it: `--unreal-cvar <name>=<number>` or `=x<factor>`.
- "3D resolution" in the launcher (`--unreal-render-scale <percent>`, `run-windows.ps1 -RenderScale`)
  scales the size the game renders its 3D at; the game's upscaler still makes the full picture.
- Saros at one heavy spot: 31 fps as it is, 33 at 75%, 36 at 50%.
- Details: [docs/unreal-settings.md](docs/unreal-settings.md).

### Occlusion queries

- **The game's occlusion queries are answered.** A game asks the graphics card how much of an
  object can be seen and stops drawing what is hidden. The emulator answered "visible" every
  time unless it was started with `--occlusion-queries`. The launcher (Graphics, "Occlusion
  queries") and `run-windows.ps1` now turn that option on; `-OcclusionQueries off` gives the old
  behaviour. The notes of the [Jetsku fork](https://github.com/Jetsku/KytyPS5) pointed here.
- Saros at 50% 3D resolution, at two heavy spots: 41 to 50.6 fps (+24%) in a large hall, with
  1,547 to 619 draws a frame, and 41 to 57 fps (+37%) at another, with 1,540 to 725. Where
  little is hidden the gain is small: 51.5 to 53.7 fps while turning all the way round. With
  frame generation on it is the same gain.
- **Their results no longer stop the thread that feeds the GPU** (after the Jetsku fork). They
  were written at the next label, flush or flip, after a wait for the GPU: one wait of 2 ms a
  frame in Saros. They are now written when the GPU has run them, by the thread that writes
  the labels and ahead of the labels that follow: 50.1 to 51.2 fps in three off/on rounds
  (`occlusion_defer`).
- Checked: the still picture is the same, and captures taken while the camera turned, and
  right after a fast turn, miss nothing. Not judged by eye in motion. That fork turned its own
  queries off by default: they cost half the frame rate in the heaviest levels of another game.

### Shaders compiled before play instead of during it

- **Precompile pass** (`run-windows.ps1 -Precompile`, or the launcher's button): builds everything
  known for the game into the driver's cache and exits.
- **The pass runs by itself before the game starts when it has to** (launcher switch "Before
  playing", on by default; `run-windows.ps1 -PrecompileFirst on`). A data folder keeps a stamp
  of its last finished pass: the emulator, the game, the shader lists and the graphics driver
  it was for. A start that finds another stamp, or none, builds first and then starts the
  game. Before this, a game started in a folder whose shaders were not built compiled them
  while it ran: one such session stalled for 9.3 s in its first seven minutes, a second at the
  longest. Closing the pass's window starts the game at once.
- **The whole game's compute shaders**, taken from the game's own files (`--import-shaders`,
  `shader-seeds/`), not only the ones an earlier session met. The precompile pass reads them out
  of an Unreal Engine 5 game by itself the first time it runs (a few seconds for Saros: 37,545
  shaders from six shader libraries), and again after a game update. The test build of
  2026-11-16 could not do this yet: without a seeds file made by hand, its pass built only what
  sessions had recorded and the Saros lists.
- **Pipeline lists** (`pipeline-lists/`): recordings shared without any game code; a list for Saros
  is included. The list of 2026-11-17 has 14,603 compute programs and 977 graphics pipelines
  (12,655 and 610 before): the recordings of every session since, five starts of a new game
  among them. `-ExportPipelineList` now adds a data folder's recordings to the list instead of
  writing them alone.
- **Shader pairs** (`shader-pairs/`): vertex and pixel shaders likely to be drawn together are
  built ahead; a list for Saros is included.
- **At every start**, the recorded programs are translated on helper threads and recorded
  pipelines are built in the background before they are needed.
- **"Never wait for a new graphics shader"** (`--skip-pending-draws`): a graphics shader met for
  the first time is built in the background and its draw is left out until it is ready, instead
  of the game stopping.
- The driver's own shader disk cache is kept in the game's data folder.
- The pass's summary now also says how much earlier passes already built, so a short re-check no
  longer looks like a lost cache.
- **The pass builds on every CPU but two.** It used half of the threads and never more than
  eight. Nearly all of its time is the graphics driver's shader compiler (94 to 98% of a build
  thread with an NVIDIA card), so the gain is what the extra threads are worth: on an i9-13900K
  the same first 3,000 builds take 118 s instead of 208 s, and the first phase (the recorded,
  listed and compute shaders, 13,800 builds) 15 minutes instead of about 27. On a PC with
  little memory it takes fewer threads; `-PrecompileThreads <n>` names a count.
- **The progress line says how far the pass is**: the total, the share done and about how long
  is left at the pace of the last two minutes.
- **A pass can be stopped and continued.** Closing the emulator's window ends it; what it built
  is saved then, and every five minutes while it runs, and the next pass goes on from there.
  Before, the save after a stop could fail and lose everything the pass had built, a pass saved
  nothing until its end, and stopping twice during the same long compile shut that shader out
  of later passes.
- **"Also build likely shader pairs"** in the launcher (on by default; `-ShaderPairs none`):
  off, a first pass leaves out the shader pairs, about 23,000 of the 37,000 pipelines of Saros.
- Details: [docs/shader-precompile.md](docs/shader-precompile.md).

### Crashes and memory

- **Eight fixes of the Saros fork's `saros-v0.1.8` to `saros-v0.1.12` are taken over** (its author
  in turn took our lazy buffer synchronisation, the white flash fix and the readback work). Each
  is its own commit with his text:
  - *Lost graphics device*: a block-compressed texture could be given more mip levels than its
    host image holds, and uploads of the extra ones wrote past the image's memory; and a draw
    could keep an image that another binding of the same draw had just replaced. Both end in a
    lost device once that memory is freed.
  - *Wrong textures after a respawn* ("the rainbow outfit"): an image created again over memory
    that had just been written back was uploaded from a stale buffer.
  - *Objects missing with occlusion queries on*: a draw the emulator leaves out inside one of
    the game's queries (its pipeline is still being built, its shader cannot run, it is
    replaced by a copy) counted as hidden, so the game could cull what its draw shows. It now
    counts as visible. Unlike the Saros fork, such a draw does not wait for its pipeline here:
    it is left out as outside a query, so the frame does not stop for the build.
  - *12 GB cards*: when the video memory budget fell during a load and came back seconds later,
    the buffer collector stayed at the lower mark for the rest of the load and evicted buffers
    shaders still read.
  - *A crash*, `failed to read color metadata backing`, after the game unmapped the memory a
    texture's metadata was in.
  - *Reports*: a crash of the game now also lists the memory around its registers, the last
    faults the emulator resolved on that thread and where every other game thread was; and a
    scan reports up to 8,191 buffer accesses where it reported 1,023.
  Not taken: what follows his move to a newer base of the main project (the pipeline record
  formats 8 and 9), his launcher's settings, the background rebuild of a stale driver cache,
  and the pieces this branch has in its own form (the device fault record, the commands of a
  lost submission, the equal-address volume fix). Tried and set aside: his selection feedback,
  which stops keeping images alive for textures a shader's table lists and never selects
  (several GiB of video memory while walking in Shattered Rise, his number). It builds here,
  but its GPU test fails on this branch's base (the test's shader samples the first texture
  of its table whatever the key, also with his new code switched off), so it stays on the
  side branch `wip/selection-feedback` until that is understood.
- **The Saros fork's fixes up to `saros-v0.1.7` are merged**: the emulator follows the driver's
  current video memory budget and frees memory at once when it is over the limit, a lost
  graphics device on the lift ride into Shattered Rise and in similar spots is fixed (an older
  copy of a picture was written over newer game data), and so is a freeze when video memory
  was nearly full. A failed wait for the GPU now names the Vulkan result and the memory heaps
  in the log, and the log warns while video memory is over its budget.
- **The shader preparation gives way where memory is short.** Translating the compute shaders
  of the whole game ahead of their first use holds 6.6 GB of RAM in a Saros session (23.1 GB
  against 16.5 GB at one spot): it is done on a PC with more than 40 GB of RAM, and left to
  the first use (about 13 ms each) on one with less; `KYTY_PREPARE_CREATED=1` or `=0` decides
  it by hand. The pipelines a session keeps ready for their first use take about 0.7 GB of
  video memory, and the background builds at the start of a session leave 0.9 to 2.6 GB
  behind in the driver: when video memory comes near its limit the kept pipelines are
  released and the builds not started yet are left undone.
- **The table that tells shaders where a buffer lives takes the video memory it needs** (after
  the DS fork). It has an entry for every 16 KiB of the game's address space and was made whole:
  768 MiB of video memory on every card, nearly all of it zeros. It now starts at 64 MiB and
  grows to the highest buffer the game registers, 336 MiB in Saros: video memory in use at a
  test spot went from 8.1 to 7.7 GB. `KYTY_BDA_TABLE_GIB=0` keeps it whole.
- **A card short of video memory does not build pipelines beside the game.** Those builds keep
  about 1.6 GB of video memory at a heavy spot (the pipelines held ready, and what the driver
  keeps of each pipeline it made), and giving them up later does not give the memory back. With
  a video memory budget under 13 GiB (cards of 12 GB and less) a session leaves them out: what a
  precompile pass built comes from the driver cache at its first use, anything else is compiled
  then. `KYTY_SESSION_PRECOMPILE=1` builds them on any card, `=0` on none.
- **A buffer is made in video memory even when the budget is used up**, and in system memory
  only when the card has none left. Before, the allocator put it in system memory without a
  word once the budget was reached, where the GPU reads it over the bus at every draw until the
  game is restarted (seen on the DS fork). The log warns when it happens all the same.
- The allocator takes video memory in blocks of 64 MiB instead of 256 MiB on cards with less
  than 13 GiB (from the DS fork, where it gave 2 to 6% on a card of 8 GB; not measured here).
- **A session no longer writes the shader cache again when nothing was added to it.** After a
  precompile pass every session wrote the whole file twice (2 GB for Saros, and as much memory
  for the copy while it did).
- **The log names the PC's memory**: RAM, free RAM and what can still be committed, at the
  start and again with a fatal error. Running out of memory shows as many things (a lost
  graphics device, a crash inside the game); a report can now be checked for it.
- **Much RAM is not much memory to spare, and the emulator now looks at the second.** A
  session here lost the graphics device three minutes into play on a PC with 64 GB: it had
  started with 18.5 GB that Windows could still give out (a web browser alone held 36 GB in 192
  processes) and stopped holding 23.6 GB with 0.6 GB left. Three things follow. The shader
  preparation that holds 6.6 GB steps back when less than 28 GB can be committed at the start,
  not only on a PC with little RAM (so that session would have held about 17 GB). A start with
  less than 24 GB to spare says so (`WARNING: Windows can give out only ...`) in the console
  and the launcher's log. And a fatal error with less than 2 GB left says that Windows was out
  of memory when it happened, the likely reason.
- **A fatal error where the game reuses the address of a 3D render target is fixed** (from a
  player's report, `unresolvable equal-address image overlap`). Saros places a deeper 3D render
  target of another width where a smaller one was, with the same row length; the emulator knew
  the deeper one only when the width was the same. The larger image now takes over what the
  smaller one holds. The message of the cases still not handled names both images' sizes.
- **A lost graphics device leaves a record.** With the driver's `VK_EXT_device_fault` the log,
  and the launcher's log, say how full video memory was and what the driver recorded: the
  address the GPU faulted on, or none (the GPU stopped answering and Windows reset it).
  `KYTY_DEVICE_FAULT=0` leaves the extension off. An address is looked up among the emulator's
  buffers, the live ones and the last 65,536 destroyed, and the record says whether an
  upscaler's commands were in the lost submission. Two records so far: a player's (a write
  through an invalid address, hours into a session) and one here (a read through an address of
  no buffer, during the load into a save, with DLSS upscaling on).
- **The record also says what the last twelve submissions held**: how many guest dispatches and
  the shaders of the last six, render pass instances, buffer uploads, and whether texture
  uploads, downloads or the upscaler were in them. The failing call names the submission the
  GPU did not finish, so the two together say what it was doing. The loss here came back in a
  loop of test starts with DLSS upscaling on, twice in 62 and once before the loop, always a
  read at about the same low address that is in none of the emulator's buffers, during the
  load or at the arrival in the level; none of those starts had this record yet, and the cause
  is not known.
- **A start that dies inside a driver says where.** A player's log of a start with frame
  generation ended in the middle of the Vulkan device's creation, without an error: the
  process had crashed inside a library that is not the emulator's, before the emulator's crash
  report is set up, and the file's last 4 KiB were never written. Now a log begins with the
  build, says `Creating the Vulkan device` (through NVIDIA Streamline, with DLSS upscaling
  asked for) before the call, is written out at least once a second and at every message of
  Streamline, and an exception nobody handles is named from the process' first moment:
  `--- Crash ---` with its code and the module it happened in.
- **The crash note names the callers.** The same player's next start, with frame generation and
  DLSS upscaling, ended with that note: an access violation inside the Vulkan loader
  (`vulkan-1.dll`, a read through an invalid pointer) while the device was made. In that loader
  version the address is the check of a device handle (`loader_get_icd_and_device`): the loader
  was handed a device that is not one it made, before the emulator had the new device back. The
  emulator's request and every line of Streamline up to there are the same as in a start that
  works here, so the handle came from what runs inside the call on that PC: NVIDIA's runtime
  asking for its functions, or another program's Vulkan layer (that PC has the layers of Epic's
  overlay and of RTSS, which this PC has not). One address cannot say whose call it was. The
  note now also holds the registers, the bytes of the failing
  instruction, the callers up the stack by module and offset, and the modules in the process
  that are not Windows' own (overlays and hooks of other programs show there).
  `KYTY_CRASH_TEST=1` ends a start with a crash made on purpose, to check the note on any PC.
  The log also says which Vulkan extensions NVIDIA's NGX asked for.
- **The graphics driver gets 64 GB of address space where it had 16.** With the layers left out
  (next entry), that player's start no longer crashed: the log now says what was underneath,
  `Could not create the Vulkan device ... ErrorInitializationFailed`. The NVIDIA driver makes
  the device that DLSS and frame generation ask for only when it finds address space below
  1 TB, and the emulator, which reserves the game's address space up to 1008 GB, left it the
  16 GB above that. Measured on an RTX 4090: the driver reserves two blocks there, each as
  large as it finds. Of 16 GB it gets one block of 8.5 GB and no second, of 64 GB 34 + 17 GB.
  One block is enough for that card; an RTX 5070 on a newer driver was refused. With DLSS or
  frame generation asked for, the game's address space now ends at 960 GB (the game maps
  nothing above 832 GB) and the 64 GB above it are kept free for the driver until the device
  is made. `KYTY_DRIVER_SPACE_GB=<16 to 128>` chooses another size. The log says how much was
  free before the call and how much the call took (`Address space below 1 TB ...`), and the
  error message of a refused device carries the numbers. Not shown yet on the card that was
  refused.
- **With frame generation, other programs' Vulkan layers are left out.** The overlays of Steam,
  Epic, GOG Galaxy and RTSS (MSI Afterburner) and OBS's game capture each put a layer into every
  game, and with frame generation they run between NVIDIA's runtime and the driver. In the
  start above the driver had refused the device, and the crash in the loader is what a layer
  going on with a device that was never made looks like: with the layers left out the same
  start ends with the driver's refusal in plain words. Leaving them out is the
  loader's own switch (`VK_LOADER_LAYERS_DISABLE`, with the graphics vendors' own layers kept),
  set by the emulator when frame generation is on and the environment names no filter itself.
  The log says `Vulkan layers left out` when it applies. What it costs: those overlays and
  OBS's game capture do not show in a session with frame generation (OBS's window capture
  uses no layer). `KYTY_VULKAN_LAYERS=1` keeps the layers; `=0` leaves them out without frame
  generation too.
- **The emulator's log is written by default.** The launchers' Log setting and
  `run-windows.ps1` without `-Log` now choose the file (`logs\<time>.log` in the data folder),
  so a report of a crash has the log without a second try. Measured at one spot of Saros: no
  frame rate lost (56 fps with the log, 53 to 54 without), but the start into the save took
  about 7 s longer (42 and 47 s against 35 and 39), and one such start writes some 25 MB. The
  run script therefore keeps the five newest of these files and removes older ones. Log: Off
  in the launcher (`-Log off`) is as before; a settings file that says `off` keeps saying so.
- With 12 GB of video memory or 16 GB of RAM Saros is still at the edge; a lower 3D resolution
  and no frame generation leave more room.

### Smaller things

- **A white flash every ten seconds or so is gone.** For one frame the scene lost its
  ray-traced shadowing and showed raw, speckled, over-bright lighting, every 608 frames. A
  shader that walks the game's ray tracing structure read "no node here" from a page of it
  while the buffer cache was watching that page (it does so to learn whether the page is still
  read), so every ray hit nothing for the frame or two the watch lasted. The watch is as old as
  the Saros fork; this branch made it regular. Every build of this branch had the flash.
- `KYTY_HOST_GAMEPADS=0` makes the emulator ignore the PC's gamepads.
- A fatal error ends the process at once instead of hanging in library shutdown.
- The emulator opts out of Windows power throttling.

## Performance

At a heavy spot of Saros (about 1,560 draws, 1,120 indirect draws and 830 dispatches a frame, no
frame generation) the branch went from 29.7 fps to about 35 fps at the game's own 3D resolution
during this work, and runs at about 39 to 40 fps at 50% 3D resolution. At a lighter spot the
unmodified `saros` branch gave 37 fps and the first batch of changes below 47.9.

### The thread that feeds the GPU

- **Vulkan commands are recorded on a thread of their own**: 38.0 to 40.9 fps.
- **Buffers are synchronized only where shaders read them through the page table**: 40.4 to
  47.9 fps at the lighter spot, the largest single step. Guest write faults fell from about
  3,700 to 840 a frame.
- **Draws that bind the same small constant buffer share one copy of it** (idea from the DS
  fork): copies 10,900 to 2,600 a frame, 2.6 ms less CPU a frame, about +2.3 fps at 50%.
- **Saros's main texture table stays resident on the GPU**: the table has about 200,000 entries
  and 2,100 textures, and was rebuilt at every draw that reads it. About +1 fps at 100% and
  +3.4 fps at 50%.
- **The look-ahead that prepares resource tables on helper threads plans again where the command
  stream leaves what it scanned** (command lists the game writes on the GPU): table walks left to
  the GPU thread fell from 226 to 53 a frame, 34.1 to 35.1 fps.
- **No barrier between storage writes that draws repeat** (from the DS fork): 30.2 to 31.1 fps.
- **A read that no pending label can touch skips the label list's lock**: about 1 ms less CPU a
  frame.
- **Address ranges are kept in blocks of sorted intervals instead of a tree** (from the DS fork):
  the sets of GPU-written, pending and mapped ranges took 3.3% of the GPU thread at the heavy
  spot and take 1.1%; 24.3 to 23.6–24.0 ms of that thread a frame at 50%. The frame rate moved
  within what two runs differ by (40.4–40.7 before, 40.1–41.6 after).
- **A GPU-driven draw or dispatch that does nothing is left out when that is known without a
  wait** (`skip_empty_indirect`, off by default, see `fault_ahead` below; the fork of Jetsku
  skips empty indirect dispatches the same way). It rarely is in Saros, and that is the finding: at the heavy spot 170 of 310 GPU-driven
  draws a frame draw no instance and 180 of 420 indirect dispatches run no workgroup, each
  with the full preparation of a draw or dispatch, but their arguments are written by a culling
  pass recorded just before them and are still with the GPU when the packet comes. Two or
  three draws a frame are found without waiting. Waiting for the arguments finds all of them
  (draws 604 to 433 a frame, dispatches 812 to 673) and costs more than it saves: seven more
  waits a frame, 49.5 to 45.0 fps (`skip_empty_indirect 2`, for measuring).
- **An option that guesses which GPU-driven draws are empty** (`predict_empty_draws`, off by
  default; `KYTY_PREDICT_EMPTY_DRAWS=1` in the launcher's Environment field turns it on). It is
  not exact: a draw whose record the GPU still holds is left out when the same draw (same
  shaders, index buffer, targets, viewport and place in the frame) had nothing to draw the last
  four times it was checked and nothing in the last 90 frames. Every such draw's record is
  still read each frame, by a copy at the draw's place in the command stream that arrives
  without a wait, so a draw that gets something to draw is recorded again a frame or two
  later. At the heavy spot about 125 of the 170 empty draws a frame are left out, draws 604
  to 482, the render thread's own work 16.1 to 14.4 ms a frame; about one draw in ten frames
  is left out wrongly, standing, turning or walking, and stills with it on and off do not
  differ. Most of the saved time comes back as waiting for the GPU: 50.0 to 51.3 fps in one
  session, 50.5 to 51.2 in another.
- The GPU thread runs above the game's own threads.

### Readbacks (the game reading what the GPU wrote)

- **A game thread finishes its own readback** instead of waiting for the GPU thread: 29.6 to
  31.3 fps, and the slowest 5% of frames from 49.6 to 33.8 ms.
- **Only the pages a store or a table walk asks for are read back**: 38.8 to 40.3 fps.
- **Readbacks over 64 KiB go through the download buffer**: 40.7 to 41.6 fps, worst frame 66 to
  34 ms.
- In-place readbacks copy with streaming loads.
- **A value only the shader takes is copied to it on the GPU instead of read back** (the DS
  fork's direction for GPU-written draw arguments, here for the values a resource walk loads).
  Saros chains small compute passes, each loading a few bytes the one before wrote, and the
  thread that feeds the GPU stopped for every such load: 8 waits a frame at the heavy spot.
  The walker now knows which loaded values nothing on the host depends on; for those a
  compute dispatch copies the bytes into its own data on the GPU. Waits 6.9 to 5.0 a frame,
  35.0 to 35.6 fps in four off/on rounds on a busy PC (`gpu_scalars`). The waits that remain
  are values the host needs itself: two counts that go into a buffer's address, the thread
  counts of two indirect dispatches, and a command list the game builds on the GPU.

### The game's stores into tracked memory

- **A store that walks through tracked pages opens them a window at a time** (after the
  fault-ahead of the Jetsku fork). Saros took about 1,700 write faults a frame on pages the
  buffer cache had write-protected, three in four of them beside another one: game threads
  filling a buffer page after page, each page an exception and a protection change, and
  another protection change when the page is uploaded. A thread that faults on the page after
  its last fault now gets the following pages opened with it, 2, 4, 8 and then 16 at a time; a
  single fault opens nothing, so scattered stores cost no extra uploads. The window stops at a
  page the GPU owns and at one an image watches too. At the heavy spot with DLSS at 1440p:
  faults 1,720 to 585 a frame, protection changes 2,730 to 1,785, the emulator's CPU time 111
  to 104 ms a frame, buffer uploads +3%. The frame rate there is held by the thread that feeds
  the GPU and moved from 51.8 to 52.2 fps; the picture is the same (`fault_ahead`: the most
  pages a fault opens, 0 for none). **Off by default** (`KYTY_FAULT_AHEAD=16` turns it on): in
  the loop of test starts above, 2 of 26 lost the graphics device with this and
  `skip_empty_indirect` on against 0 of 26 with one or both off. That neither convicts nor
  clears them, the same loss was seen once before either existed, and neither makes a frame
  faster, so they wait for the cause.

### Textures

- **A rendered image is no longer uploaded again from stale memory when a small unrelated buffer
  is written at its address** (Unreal reuses one block of memory for several resources in a
  frame): image uploads 320 to 130 MiB a frame, about +1 fps at 100%.

### Shader translation

- **Compute programs whose 64 lanes do not depend on each other run one lane per invocation**
  (from the DS fork): 357 of Saros's 563 such programs qualify; GPU time -1.7%.
- **One module per compute shader serves whatever the game binds**: buffer formats, strides,
  texture channel orders and level counts are read at run time, which is what lets the whole
  game's compute shaders be built before play.

### Switches for comparing

Most of the changes above can be turned off while the game runs, through the live channel below:
`vulkan_recording`, `bda_lazy`, `stream_share`, `heap_by_content`, `lookahead_replan`,
`guest_image_writes`, `image_buffer_writes`, `guest_copy`, `readback_narrow`, `gpu_scalars`,
`fault_ahead`, `game_upscaler`, `game_upscaler_skip`. Each also reads an environment variable `KYTY_<NAME>` at
start. `present_crop` 1 to 9 shows a part of the picture pixel for pixel in the window.

## Tools for measuring

- **Live channel** (`KYTY_LIVE_FILE=<file>`): commands written to a file while the game runs.
  `measure <seconds>` reports fps, frame times, GPU time and per-frame counters; `prof` samples a
  thread and can look inside one function; `set <switch> <value>` flips a switch; `mem` searches,
  reads and writes the game's memory; `pad` and `stick` send controller input. `measure`
  switches the performance panel's counters on for its window; `fps <seconds>` reports the
  frame rate alone and leaves them as they are (off and on gave the same rate on Saros).
- **A performance log of a whole play session** (`KYTY_PERF_LOG=<seconds>`, for example
  `KYTY_PERF_LOG=10` in the launcher's Environment field): every such window of play writes one
  `Perf <time of day> ...` line with the numbers of `measure` into the log, so a session played
  through any part of the game says afterwards where it was slow and whether the CPU or the GPU
  was the limit there. Windows without a frame (loading) are left out.
- **The GPU's time by shader** (live switch `gpu_prof`; the idea of the Jetsku fork's operation
  profiler). While it is on, a timestamp is recorded before and after every guest dispatch and
  render pass instance, and turning it off writes `GpuProf` lines into the log: the GPU's time a
  frame for each compute shader (by its hash, with how often it ran), for draws, buffer
  uploads, texture uploads, downloads and the upscaler. The timestamps make the commands run
  one after the other and cost about 3 µs each, so the frame rate drops while it runs; the
  shares are what it is for. At the heavy spot of Saros with DLSS at 1440p it says: compute
  shaders about 60% of the GPU's time, spread over 350 of them with the largest at 5%, the
  upscaler 8%, draws 7%, uploads and downloads together about 11%. It also settled a guess:
  leaving out the 440 compute-to-all barriers a frame (a test build) changed neither the GPU's
  time nor the frame rate, so barriers are not what the GPU waits on.
- **A card with less video memory, for tests on a bigger one** (`KYTY_SIMULATE_VRAM_MB=<n>`, from
  the DS fork): the rest of the card's memory is taken at start and kept, so the driver and
  Windows treat the game as on a card of n MiB.
- **An upload worker** for buffer synchronization exists behind `async_upload` and is off: it did
  not pay on Saros.
- `build-windows.cmd` can build with profiling symbols and with a locally trained PGO profile.

## Not done, and known limits

- Table mode of the DS fork is not ported: as written it takes only draws without bound vertex
  buffers, which is 3% of Saros's draws. Its ideas that carry over are in the list above.
- The thread that feeds the GPU still waits about 5 times a frame at a heavy spot for bytes the
  GPU writes and the host needs itself, and time saved before those waits comes back as longer
  waits. An upload worker for the per-draw uploads (the DS fork's scope) cut that thread's
  copies and page protections by 5% of its time and changed no frame rate for that reason; it
  is not in.
- The DS fork's merged page protections are not ported: they join protections of adjacent
  pages that follow each other, of which Demon's Souls has about 110 a frame and Saros one.
- Cards of 12 GB and less are not confirmed. With the rest of a 24 GB card taken away
  (`KYTY_SIMULATE_VRAM_MB=12288`, on a desktop whose other programs held 5 GB of it) Saros ran
  at 20 fps at 100% 3D resolution and at 52 fps at 50%, where the whole card gives 53 and 60:
  Windows kept 1.3 to 2 GB of the game's video memory in system memory.
- Three reports of a lost graphics device are not explained: one on the first test build, two on
  the build of 2026-11-17c (an RTX 4070 with frame generation, less than a minute into a
  session, with 8.4 GB of its 11.8 GB video memory budget in use; another PC without frame
  generation about ten minutes in, whose log names neither the card nor its memory). Not
  reproduced here: two boots with frame generation on a 24 GB card with 8 to 9 GB of it taken
  away ran without one. The next report will carry the driver's fault record (see above).
- With DLSS upscaling on, one of about twenty-five test boots lost the graphics device during
  the load into a save (the record above); four more boots of the same setting did not. Not
  explained, and not known to be DLSS's doing: the next record says which submission held
  DLSS's commands.
- A shader that reads the resident texture table is another compiled variant than before. It is
  built at first use in each part of the game and cached from then on.
- CPU writes into memory that a rendered image also uses still cause whole-image uploads (about
  75 MiB a frame at 100% on Saros).
- Frame generation, the launcher app and the run scripts are Windows only, and so is reading the
  shaders out of a game's files: `ooz_decode`, the program that unpacks Unreal's shader
  libraries for it, is built for Windows only. It is a separate program under GPL-3.0-or-later
  (the ooz decompressor by Powzix), started by the precompile pass and never linked with the
  emulator (`tools/ooz-decode/README.md`).
