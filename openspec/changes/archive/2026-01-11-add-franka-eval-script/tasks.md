# Tasks: Add Franka Robot Evaluation Script

## Phase 0: Prerequisites

- [x] **T0.1** Create `src/openpi/policies/franka_policy.py` with `FrankaInputs` and `FrankaOutputs`
  - Follow `droid_policy.py` pattern for transform structure
  - Map observation keys: `observation/image` → `base_0_rgb`, `observation/wrist_image` → `left_wrist_0_rgb`
  - Handle 14D dataset state → extract first 7D (TCP pose) for model's `state` input
  - Output 8D actions [x, y, z, qw, qx, qy, qz, gripper]
  - Add `make_franka_example()` for testing

## Phase 1: Core Infrastructure

- [x] **T1.1** Create `examples/franka/` directory structure
- [x] **T1.2** Create `examples/franka/constants.py` with default robot/camera/control constants
- [x] **T1.3** Create `examples/franka/real_env.py` with `FrankaRealEnv`
  - Adapt robot integration from `reactive_diffusion_policy`
  - Implement raw robot state collection
  - Implement action execution with velocity limiting and workspace clipping
- [x] **T1.4** Create `examples/franka/camera_service.py` (Python 3.9) with IPC
  - Serve latest L500/D400 RGB frames over a length-prefixed msgpack protocol
  - Use a provider interface to plug in RDP camera drivers
- [x] **T1.5** Create `examples/franka/camera_client.py` for Python 3.9 camera service
  - Connect to the external camera service (RDP stack)
  - Fetch L500/D400 RGB frames with timeout handling
- [x] **T1.6** Create `examples/franka/env.py` with `FrankaEnvironment`
  - Implement `openpi_client.runtime.environment.Environment`
  - Format observations to match `FrankaInputs` expectations (observation/image, observation/wrist_image, observation/state)
  - Apply per-step actions (chunking handled by ActionChunkBroker)

## Phase 2: Main Script

- [x] **T2.1** Create `examples/franka/main.py` entry point
  - Support loading checkpoints via `policy_config.create_trained_policy()`
  - Parse command-line args with tyro
  - Support both local inference and remote server modes (both synchronous)
  - Use `Runtime` + `PolicyAgent` + `ActionChunkBroker`
  - Write an episode summary (CSV/JSON) to `output_dir`

## Phase 3: Documentation and Testing

- [x] **T3.1** Create `examples/franka/README.md` with:
  - Prerequisites (C++ franka_control server, camera setup)
  - Usage examples for both local and remote modes
  - Configuration options
  - Safety guidelines

- [ ] **T3.2** Manual test on real robot (example: `./checkpoints/11999`)
  - Verify observation format matches training data
  - Verify action format and dimension (8D: position[3] + quaternion[4] + gripper[1])

## Dependencies

- T0.1 is a prerequisite for all other tasks (config.py imports franka_policy)
- T1.3 depends on T1.1, T1.2
- T1.4 depends on T1.1 (can run in parallel with T1.3)
- T1.5 depends on T1.4
- T1.6 depends on T1.3, T1.5
- T2.1 depends on T0.1, T1.6
- T3.1 can run in parallel with T2.1
- T3.2 depends on T2.1

## Verification

Each task should be verified by:
- T0.1: `python -c "from openpi.policies import franka_policy; print('OK')"` succeeds ✅
- T1.x: Code compiles, imports work ✅
- T2.1: Script runs with --help ✅
- T3.1: Documentation is complete and accurate ✅
- T3.2: End-to-end evaluation runs successfully on real robot (pending hardware access)
