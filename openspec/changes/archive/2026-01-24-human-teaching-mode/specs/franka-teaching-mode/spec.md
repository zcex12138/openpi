# Franka Teaching Mode Specification

## ADDED Requirements

### Requirement: Teaching Mode Activation
The system SHALL switch to zero-stiffness impedance control when the user presses the spacebar during evaluation, allowing the operator to manually guide the robot.

#### Scenario: Spacebar triggers teaching mode in impedance control
- **WHEN** the evaluation is running in impedance control mode
- **AND** the user presses the spacebar
- **THEN** the system SHALL set translational and rotational stiffness to 0.0
- **AND** the robot SHALL become compliant and guidable by hand

#### Scenario: Spacebar ignored in non-impedance control mode
- **WHEN** the evaluation is running in cartesian control mode
- **AND** the user presses the spacebar
- **THEN** the system SHALL log a warning
- **AND** the control mode SHALL remain unchanged

#### Scenario: Idempotent activation
- **WHEN** teaching mode is already active
- **AND** the user presses the spacebar again
- **THEN** the system SHALL take no action
- **AND** no error SHALL be raised

### Requirement: Continuous Inference During Teaching
The system SHALL continue running policy inference and updating the ImpedanceMotion target while in teaching mode, even though zero stiffness means the target updates produce no control force.

#### Scenario: Inference continues after teaching mode activation
- **WHEN** teaching mode is activated
- **THEN** the policy inference loop SHALL continue executing
- **AND** `execute_action()` SHALL continue updating `ImpedanceMotion.target`
- **AND** the step counter SHALL continue incrementing

### Requirement: End-Effector Load Configuration
The system SHALL call `robot.set_load()` when entering teaching mode to configure gravity compensation for the end-effector payload.

#### Scenario: Load set on teaching mode entry
- **WHEN** teaching mode is activated
- **THEN** the system SHALL call `robot.set_load()` with configured mass, COM, and inertia
- **AND** if `set_load()` fails, the system SHALL log a warning and continue

#### Scenario: Load parameters from configuration
- **GIVEN** the configuration file contains `teaching.load_mass`, `teaching.load_com`, and `teaching.load_inertia`
- **WHEN** teaching mode is activated
- **THEN** the system SHALL use these configured values for `set_load()`

### Requirement: Recording Continuity
The system SHALL continue recording frames without interruption when switching to teaching mode.

#### Scenario: Recording persists through mode switch
- **WHEN** teaching mode is activated
- **THEN** the `EpisodePklRecorder` SHALL continue recording frames
- **AND** if the recorder queue is full, frames MAY be dropped with a warning

### Requirement: Teaching Segment Marking
The system SHALL mark each recorded frame with an `is_human_teaching` boolean field indicating whether the frame was captured during teaching mode.

#### Scenario: Frames marked correctly before and after switch
- **WHEN** teaching mode is activated at step N
- **THEN** frames before step N SHALL have `is_human_teaching = false`
- **AND** frames from step N onward SHALL have `is_human_teaching = true`

### Requirement: One-Way Teaching Switch
The system SHALL maintain teaching mode until the episode ends via Ctrl+C. Re-pressing spacebar SHALL NOT restore automatic control.

#### Scenario: Teaching mode persists until episode end
- **WHEN** teaching mode is active
- **AND** the user presses any key including spacebar
- **THEN** teaching mode SHALL remain active
- **AND** only Ctrl+C SHALL end the episode

#### Scenario: Timeout disabled after teaching mode
- **WHEN** teaching mode is activated
- **AND** the elapsed time exceeds `max_episode_time`
- **THEN** `is_episode_complete()` SHALL return false
- **AND** the episode SHALL continue until Ctrl+C

### Requirement: Teaching State Reset
The system SHALL reset the teaching mode state to false at the beginning of each new episode.

#### Scenario: Teaching state reset on episode start
- **GIVEN** teaching mode was active in the previous episode
- **WHEN** a new episode starts via `reset()`
- **THEN** `is_teaching_mode` SHALL be false
- **AND** normal impedance control SHALL be active

### Requirement: Non-TTY Environment Handling
The system SHALL gracefully degrade when stdin is not a TTY, disabling keyboard detection and logging a warning.

#### Scenario: Keyboard disabled in non-TTY environment
- **WHEN** `sys.stdin.isatty()` returns false
- **THEN** the system SHALL log a warning about keyboard teaching being disabled
- **AND** spacebar detection SHALL be skipped
- **AND** all other functionality SHALL work normally
