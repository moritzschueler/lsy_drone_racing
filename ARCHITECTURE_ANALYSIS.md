# LSY Drone Racing: Complete Architecture Analysis

## 1. Main Environment Class

### **`RaceCoreEnv`** ([race_core.py](lsy_drone_racing/envs/race_core.py#L1))

The core environment that manages drone racing simulations. Key responsibilities:

- **Physics Simulation**: Wraps MuJoCo simulation via Crazyflow's `Sim` class for realistic drone dynamics
- **Track Management**: Loads gates and obstacles from TOML configs, manages their positions
- **Gate Progression Tracking**: Monitors which gates the drone has passed, determines target gates
- **Collision Detection**: Detects contacts between drones and gates/obstacles
- **Observation Management**: Provides drone state (position, velocity, orientation) and track information
- **Reward Computation**: Calculates rewards based on gate passages and progress
- **Vectorized Execution**: Supports parallel simulation of multiple environments using JAX
- **Disturbance Modeling**: Applies randomized disturbances to actions and dynamics for robustness

### Subclasses:
- **`DroneRaceEnv`** ([drone_race.py](lsy_drone_racing/envs/drone_race.py)): Single-agent racing (Gymnasium-compatible wrapper)
- **`MultiDroneRaceEnv`** ([multi_drone_race.py](lsy_drone_racing/envs/multi_drone_race.py)): Multi-agent racing with multiple concurrent drones
- **`RealDroneRaceEnv`** ([real_race_env.py](lsy_drone_racing/envs/real_race_env.py)): Real-world deployment using Vicon mocap and ROS

---

## 2. Control/Controller Classes

### **Base Class: `Controller`** ([controller.py](lsy_drone_racing/control/controller.py))

Abstract base class defining the controller interface:

```python
def __init__(obs, info, config)           # Initialize with track/drone info
def compute_control(obs, info) -> action  # Main control loop (required)
def step_callback(action, obs, ...)       # Called after each step
def episode_callback()                    # Called after each episode
def render_callback(sim)                  # Optional visualization
```

**Control Modes**: Actions can be either:
- **State commands**: `[x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]` (13 DOF)
- **Attitude commands**: `[roll, pitch, yaw, thrust]` (4 DOF)

### **Concrete Controller Implementations:**

#### 1. **`StateController`** ([state_controller.py](lsy_drone_racing/lsy_drone_racing/control/state_controller.py))
- **Strategy**: Predefined cubic spline trajectory through hardcoded waypoints
- **Key Components**:
  - 10 hardcoded waypoints as control points
  - 15-second total trajectory duration
  - Uses `scipy.interpolate.CubicSpline` for smooth interpolation
  - Evaluates trajectory at each timestep to get desired position
  - Returns position + zero velocity/acceleration commands
- **Use Case**: Simple baseline for state-control mode
- **Limitations**: Fixed trajectory, not adapted to track layout

#### 2. **`AttitudeController`** ([attitude_controller.py](lsy_drone_racing/lsy_drone_racing/control/attitude_controller.py))
- **Strategy**: PID-based position tracking with collective thrust + attitude output
- **Key Components**:
  - Same 10-point cubic spline as StateController
  - Computes position error relative to desired position
  - PID controller gains: `Kp=[0.4, 0.4, 1.25]`, `Ki=[0.05, 0.05, 0.05]`, `Kd=[0.2, 0.2, 0.4]`
  - Gravity-compensated thrust calculation
  - Attitude computed via inverse dynamics
- **Use Case**: Attitude control mode with passive trajectory tracking

#### 3. **`AstarController`** ([astar_controller.py](lsy_drone_racing/lsy_drone_racing/control/astar_controller.py))
- **Strategy**: A* path planning to avoid obstacles, then spline-based trajectory following
- **Key Components**:
  - `astar_3d` algorithm ([astar.py](lsy_drone_racing/lsy_drone_racing/control/astar.py)): 26-connected voxel-based pathfinding
  - Voxel grid discretization for collision detection
  - Clearance margin around obstacles (0.22m detour margin)
  - Gate corner detection and waypoint optimization
  - Multiple PID trackers:
    - `pid_pos`: Position error tracking (Kp=0.5, Ki=0.01, Kd=0.40)
    - `pid_vel`: Velocity error tracking (Kp=0.25, Ki=0.01, Kd=0.1)
    - `pid_acc`: Acceleration error tracking (Kp=0.01)
  - Cubic spline trajectory from planned path
  - Look-ahead distance for smoother tracking
- **Use Case**: Obstacle-aware path planning with efficient navigation
- **Planning Cycle**: Re-plans path to next gate on-demand

#### 4. **`AttitudeMPC`** ([attitude_mpc.py](lsy_drone_racing/lsy_drone_racing/control/attitude_mpc.py))
- **Strategy**: Model Predictive Control using ACADOS solver
- **Key Components**:
  - Symbolic drone dynamics from `drone_models` package
  - Nonlinear MPC with 10+ step horizon
  - ACADOS template-based solver
  - Cost function: tracking error + input rate penalties
  - Attitude control interface (thrust + RPY)
  - Real-time receding horizon optimization
- **Use Case**: Advanced control with explicit constraint handling
- **Performance**: Computationally intensive but optimal predictions

#### 5. **`AttitudeRL`** ([attitude_rl.py](lsy_drone_racing/lsy_drone_racing/control/attitude_rl.py))
- **Strategy**: Deep RL policy (PPO) trained via reinforcement learning
- **Key Components**:
  - Loads pre-trained PyTorch model (`ppo_drone_racing.ckpt`)
  - Observation stacking (2 timesteps)
  - Input normalization
  - Neural network forward pass for action generation
  - Collective thrust + attitude output
- **Use Case**: Learning-based control without explicit modeling
- **Training**: Uses `train_rl.py` for policy training

#### 6. **`AttitudeInput`** ([attitude_input.py](lsy_drone_racing/lsy_drone_racing/control/attitude_input.py))
- **Strategy**: Interactive keyboard/joystick control
- **Key Components**:
  - Real-time input from game controller or keyboard
  - Direct attitude + thrust commands
- **Use Case**: Manual piloting and testing

---

## 3. Main Entry Points

### **Simulation (`scripts/sim.py`)**

```python
simulate(config="level0.toml", controller=None, n_runs=1, render=False)
```

**Execution Flow**:
1. Load configuration file (TOML)
2. Load controller class dynamically from specified Python file
3. Create environment with `gymnasium.make("DroneRacing-v0", ...)`
4. For each episode:
   - Reset environment → get initial observation
   - Instantiate controller with `Controller(obs, info, config)`
   - Loop until episode ends:
     - Compute action: `action = controller.compute_control(obs, info)`
     - Step environment: `obs, reward, terminated, truncated, info = env.step(action)`
     - Call controller callback: `controller.step_callback(...)`
     - Render if enabled
   - Call `controller.episode_callback()`

**Key Configuration**:
- `config.controller.file`: Which controller to use
- `config.env.freq`: Environment step frequency (50 Hz)
- `config.env.control_mode`: "state" or "attitude"
- `config.sim.render`: Enable visualization
- `config.env.track`: Track layout (gates, obstacles, drone start position)

### **Real Deployment (`scripts/deploy.py`)**

```python
main(config="level2.toml", controller=None)
```

**Differences from Simulation**:
- Creates `RealDroneRaceEnv` instead of `DroneRaceEnv`
- Integrates with ROS for Vicon mocap tracking
- Real-time loop maintains 50 Hz frequency
- Logs track completion time
- Graceful error handling for real hardware

### **Other Entry Points**:
- `scripts/evaluate.py`: Batch evaluation of controllers
- `scripts/multi_sim.py`: Multi-drone simulation
- `scripts/multi_deploy.py`: Multi-drone real deployment
- `scripts/check_track.py`: Visualize track layout

---

## 4. Execution Flow

### **Complete Execution Sequence**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION (happens once)                             │
│ ├─ Load config (TOML) → frequencies, track layout, control  │
│ ├─ Create RaceCoreEnv with Crazyflow Sim                    │
│ │  ├─ Initialize JAX/MuJoCo physics engine                  │
│ │  ├─ Load gates + obstacles into simulation                │
│ │  ├─ Build JAX-compiled reset/step functions               │
│ │  └─ Create EnvData struct with tracking state             │
│ ├─ Register environment to Gymnasium                        │
│ └─ Load controller class from Python file                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. EPISODE RESET (happens at start of each episode)         │
│ ├─ env.reset() calls RaceCoreEnv._reset(data, seed)        │
│ │  ├─ Randomize drone initial position/orientation         │
│ │  ├─ Randomize gate/obstacle positions (if enabled)       │
│ │  ├─ Reset contact flags and gate tracking state          │
│ │  ├─ Reset step counter                                    │
│ │  └─ Return initial observation + info                    │
│ ├─ controller = ControllerClass(obs, info, config)         │
│ │  └─ Controller runs initialization (e.g., plan path)      │
│ └─ Set timestep counter i = 0                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CONTROL LOOP (repeats at env frequency, ~50 Hz)          │
│ ├─ Timestamp: t = i / freq                                  │
│ ├─ Compute action:                                          │
│ │  └─ action = controller.compute_control(obs, info)       │
│ │      (generates [13] or [4] DOF command depending mode)   │
│ ├─ Step environment:                                        │
│ │  └─ obs, reward, terminated, truncated, info =           │
│ │      env.step(action)  [calls RaceCoreEnv._step]         │
│ │      ├─ Apply action to sim controllers                  │
│ │      ├─ Step physics sim N times (500 Hz / 50 Hz = 10x)  │
│ │      ├─ Check collisions (contacts with gates/obstacles) │
│ │      ├─ Update visited gates and target gate             │
│ │      ├─ Compute reward (gate passage bonus, etc.)        │
│ │      ├─ Check termination (collision detected)           │
│ │      ├─ Check truncation (max steps reached)             │
│ │      └─ Generate observation dict with updated state     │
│ ├─ Call controller callback:                               │
│ │  └─ finished = controller.step_callback(                 │
│ │      action, obs, reward, terminated, truncated, info)   │
│ │      (controller can update internal state, train models) │
│ ├─ Render (if enabled):                                     │
│ │  └─ controller.render_callback(env.unwrapped.sim)        │
│ │      env.render()                                         │
│ ├─ Check exit conditions:                                   │
│ │  └─ if terminated or truncated or controller_finished:   │
│ │      break                                                │
│ └─ i += 1, loop back to "Compute action"                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. EPISODE CALLBACK (happens once at episode end)           │
│ └─ controller.episode_callback()                            │
│    (reset internal state, save/train models, etc.)          │
└─────────────────────────────────────────────────────────────┘
```

### **Key Frequencies**:
- **Simulation frequency**: 500 Hz (physics engine)
- **Environment frequency**: 50 Hz (controller/observation updates)
- **Ratio**: Physics steps 10 times per control command
- **Attitude control frequency**: 500 Hz (onboard low-level control)

### **Critical Path in Each Step** (most expensive operations):
1. **Apply action** (~1-5 ms): Register commanded state/attitude
2. **Physics simulation** (~10 ms): 10 MuJoCo steps
3. **Collision detection** (~2 ms): Check contacts with all gates/obstacles
4. **Compute observation** (~1 ms): Extract state from simulation
5. **Controller execution** (~1-50 ms): Depends on controller complexity
   - StateController: ~0.1 ms (spline evaluation)
   - AttitudeController: ~0.5 ms (PID computation)
   - AstarController: ~5 ms (path planning + tracking)
   - AttitudeMPC: ~20-50 ms (optimization)
   - AttitudeRL: ~1 ms (NN inference)

---

## 5. Planning ↔ Control Connection

### **Architecture Pattern**

The system separates planning from control but keeps them tightly integrated:

```
┌─────────────────────────────────────────────────────────────┐
│ PLANNING LAYER (offline/on-demand)                           │
├─────────────────────────────────────────────────────────────┤
│ • Path Planning: A* pathfinding, cubic spline generation     │
│ • Frequency: Triggered on controller init or on demand       │
│ • Outputs: Reference trajectory (position curve over time)   │
└─────────────────────────────────────────────────────────────┘
                          ↓
                    (Trajectory)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ CONTROL LAYER (online, real-time)                            │
├─────────────────────────────────────────────────────────────┤
│ • Trajectory Tracking: Follow reference trajectory           │
│ • Frequency: 50 Hz (matches env.step frequency)            │
│ • Inputs: Current state + reference trajectory              │
│ • Outputs: Instantaneous control commands (state/attitude)  │
└─────────────────────────────────────────────────────────────┘
```

### **Data Flow Through Controllers**

#### **StateController & AttitudeController** (Spline-based):
```
Config (gates, obstacles)
    ↓
[Controller.__init__]
    ├─ Hardcoded waypoints → Cubic spline
    └─ Store spline as self._des_pos_spline
    ↓
[Each step: compute_control()]
    ├─ Evaluate spline at current time t
    ├─ Get position derivative from spline (velocity)
    └─ Output desired state [pos, vel, acc, ...]
    ↓
[Environment applies state/attitude control]
```

#### **AstarController** (Path-aware):
```
Observation (gate positions, drone state)
    ↓
[Controller.__init__]
    ├─ Current obs + track config
    ├─ Extract gate corner positions
    ├─ Run A* 3D pathfinding → waypoints
    └─ Spline interpolate waypoints
    ↓
[Each step: compute_control()]
    ├─ Get current error vs. trajectory
    ├─ Run PID trackers (pos, vel, acc)
    └─ Output corrected attitude command
    ↓
[Optional: re-plan on gate change]
    ├─ Detect new target gate
    ├─ Re-run A* to next gate
    └─ Generate new spline
```

#### **AttitudeMPC** (Optimization-based):
```
Track + Drone Dynamics Model
    ↓
[Controller.__init__]
    ├─ Load symbolic dynamics
    ├─ Setup ACADOS OCP solver
    └─ Pre-define waypoint trajectory
    ↓
[Each step: compute_control()]
    ├─ Current state obs
    ├─ Optimize over N-step horizon → predicted controls
    ├─ Take first control step
    └─ Output attitude command
    ↓
[Next step: re-optimize with new state]
```

### **Planning Algorithms Used**

| Algorithm | Location | Purpose | Characteristics |
|-----------|----------|---------|-----------------|
| **A* Pathfinding** | [astar.py](lsy_drone_racing/lsy_drone_racing/control/astar.py) | Obstacle-aware path planning | Voxel-based, 26-connectivity, admissible heuristic |
| **Cubic Splines** | `scipy.interpolate.CubicSpline` | Trajectory smoothing | Piecewise cubic, continuous derivatives |
| **MPC Optimization** | `acados_template` | Optimal control | Nonlinear, receding horizon, real-time solver |
| **PID Control** | [astar_controller.py](lsy_drone_racing/lsy_drone_racing/control/astar_controller.py) | Trajectory tracking | Per-axis proportional-integral-derivative |

### **Re-planning Triggers**

1. **StateController / AttitudeController**: Never re-plans (fixed trajectory)
2. **AstarController**: Re-plans when:
   - Episode starts (initial path to first gate)
   - Target gate changes (drone passed a gate)
   - Obstacle detected in path (optional adaptive re-planning)
3. **AttitudeMPC**: Continuously re-optimizes at every step (closed-loop)
4. **AttitudeRL**: No explicit planning (learned end-to-end mapping)

---

## 6. Key Interfaces & Abstractions

### **Core Data Structures**

#### **`EnvData`** (Mutable state):
Tracks dynamic environment information:
```python
class EnvData:
    target_gate: Array           # Current gate drone should aim for
    gates_visited: Array         # Which gates drone has passed
    last_drone_pos: Array        # Previous position (for gate detection)
    disabled_drones: Array       # Which drones have crashed
    steps: Array                 # Episode step counter
    marked_for_reset: Array      # Vectorized reset markers
    sim_data: SimData            # Crazyflow simulation state
    gates_pos, gates_quat: Array # Current gate poses (mutable)
    obstacles_pos: Array         # Current obstacle positions
    # ... + sensor range, position limits, etc.
```

#### **`EnvSettings`** (Configuration):
```python
class EnvSettings:
    freq: int                    # Environment frequency
    max_episode_steps: int       # Episode length limit
    disturbances: dict           # Randomized perturbations
    randomizations: dict         # Track randomization functions
    device: Device               # JAX device (CPU/GPU)
    autoreset: bool              # Auto-reset vectorized envs
```

#### **Observation** (dict from `obs()` function):
```python
{
    "pos": [x, y, z],                    # Drone position
    "quat": [x, y, z, w],                # Drone orientation
    "vel": [vx, vy, vz],                 # Linear velocity
    "ang_vel": [wx, wy, wz],             # Angular velocity
    "target_gate": int,                  # Index of next gate (-1 if done)
    "gates_pos": [N_gates, 3],           # Gate positions (nominal if not visited)
    "gates_quat": [N_gates, 4],          # Gate orientations
    "gates_visited": [N_gates],          # Which gates are in sensor range
    "obstacles_pos": [N_obstacles, 3],   # Obstacle positions
    "obstacles_visited": [N_obstacles],  # Which obstacles are in sensor range
}
```

### **Controller Interface Contract**

```python
class Controller(ABC):
    def __init__(obs: dict, info: dict, config: dict):
        """
        Inputs:
          - obs: Initial state observation
          - info: Track metadata
          - config: Full configuration (track, sim, disturbances, etc.)
        Runs once per episode
        """

    @abstractmethod
    def compute_control(obs: dict, info: dict) -> np.ndarray:
        """
        Called at env.freq (50 Hz)
        Inputs:
          - obs: Current observation dict
          - info: Optional metadata
        Returns:
          - State command [13]: [x, y, z, vx, vy, vz, ax, ay, az, yaw, wroll, wpitch, wyaw]
          OR
          - Attitude command [4]: [roll, pitch, yaw, thrust]
        Must be deterministic and real-time (< 20ms)
        """

    def step_callback(action, obs, reward, terminated, truncated, info) -> bool:
        """
        Called after env.step()
        Can be used to:
          - Update internal state
          - Collect training data
          - Trigger re-planning
        Returns True to signal episode should end early
        """

    def episode_callback():
        """
        Called once at episode end
        Can be used to:
          - Reset internal counters
          - Save/train models
          - Compute statistics
        """

    def render_callback(sim):
        """Optional: Visualize controller state in simulation"""
```

### **Environment Interface** (Gymnasium standard):

```python
env = gymnasium.make("DroneRacing-v0", 
    freq=50,                    # Control frequency
    sim_config=config.sim,      # Physics parameters
    control_mode="state",       # "state" or "attitude"
    track=config.env.track,     # Gates + obstacles
    seed=0,                     # Reproducibility
)

obs, info = env.reset()
for _ in range(max_steps):
    action = controller.compute_control(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### **Configuration System** (TOML-based):

```toml
[controller]
file = "state_controller.py"    # Which controller module to load

[env]
freq = 50                        # Control frequency
control_mode = "state"           # or "attitude"
seed = -1                        # Random or fixed seed
sensor_range = 0.7               # Gate visibility range

[env.track]
[[env.track.gates]]
pos = [0.5, 0.25, 0.7]          # Gate center position
rpy = [0.0, 0.0, -0.78]         # Gate orientation (Roll, Pitch, Yaw)

[[env.track.drones]]
pos = [-1.5, 0.75, 0.01]        # Starting position
rpy = [0.0, 0.0, 0.0]           # Starting orientation

[sim]
physics = "first_principles"     # Physics model
drone_model = "cf21B_500"        # Drone model
freq = 500                       # Physics engine frequency
render = true                    # Enable visualization
```

### **Load Chain**

```
config.toml
    ↓
load_config(path) → ConfigDict
    ↓
[env setup]
gymnasium.make("DroneRacing-v0", **config.env)
    → DroneRaceEnv.__init__()
    → RaceCoreEnv.__init__()
    → Sim(...) [Crazyflow]
    → load_track(config.env.track)
    → build_reset_fn() [JAX-compiled]
    → build_step_fn() [JAX-compiled]
    ↓
[controller setup]
controller_path = config.controller.file
load_controller(controller_path) → ControllerClass
    → [dynamic import via importlib]
    → [find subclass of Controller]
controller = ControllerClass(obs, info, config)
```

---

## 7. System Dependencies & Stack

### **Core Dependencies**:
- **Crazyflow**: Physics simulation (JAX-based MuJoCo wrapper)
- **JAX**: GPU-accelerated vectorized operations
- **Gymnasium**: Standard RL environment interface
- **MuJoCo**: Physics engine (mjx = MuJoCo JAX version)
- **SciPy**: Spline interpolation, optimization
- **NumPy**: Array operations

### **Advanced Features**:
- **drone_models**: Parametric drone dynamics (various physics models)
- **acados**: Nonlinear MPC solver (for MPC controller)
- **torch**: PyTorch for RL policies
- **ROS/rclpy**: Real-world deployment with Vicon mocap
- **crazyflow.sim.visualize**: MuJoCo rendering and marker drawing

### **Code Organization**:
```
lsy_drone_racing/
├── envs/                      # Gymnasium environments
│   ├── race_core.py          # Core RaceCoreEnv (JAX-based physics loop)
│   ├── drone_race.py         # Single-agent wrapper
│   ├── multi_drone_race.py   # Multi-agent wrapper
│   ├── real_race_env.py      # Real-world with ROS
│   ├── utils.py              # Helper: gate_passed(), load_track()
│   ├── randomize.py          # Perturbation functions
│   └── assets/               # XML specs for gates/obstacles
├── control/                   # Controller implementations
│   ├── controller.py         # Abstract base class
│   ├── state_controller.py   # Spline trajectory (state mode)
│   ├── attitude_controller.py # PID + spline (attitude mode)
│   ├── astar_controller.py   # A* planning + tracking
│   ├── attitude_mpc.py       # Nonlinear MPC
│   ├── attitude_rl.py        # RL policy
│   ├── attitude_input.py     # Manual control
│   ├── astar.py              # A* 3D pathfinding
│   ├── train_rl.py           # RL training loop
│   └── *.ckpt                # Saved models
├── utils/                     # Utilities
│   └── utils.py              # load_config(), load_controller(), draw_line()
└── __init__.py               # Gymnasium registration

scripts/
├── sim.py                    # Main entry: simulate(config, controller, n_runs)
├── deploy.py                 # Real hardware: main(config, controller)
├── multi_sim.py              # Multi-drone simulation
├── evaluate.py               # Batch evaluation
└── check_track.py            # Track visualization

config/
├── level0.toml               # No randomization (perfect knowledge)
├── level1.toml               # Drone randomization
├── level2.toml               # + Gate/obstacle randomization
├── level3.toml               # + Full track randomization
└── multi_*.toml              # Multi-drone configs
```

---

## 8. Design Patterns

### **Separation of Concerns**:
1. **Physics** (Crazyflow/MuJoCo): Isolated in `Sim` object
2. **Environment Logic** (RaceCoreEnv): Gate tracking, collision detection
3. **Control** (Controller subclasses): Decision-making
4. **Configuration** (TOML files): Decoupled from code

### **JAX Compilation for Performance**:
- Reset and step functions are `@jax.jit` compiled
- Entire episodes can run on GPU without Python/NumPy overhead
- Vectorized multi-environment execution
- Trade-off: Cannot use Python control flow in JAX functions

### **Plugin Architecture**:
- Controllers are loaded dynamically via `load_controller()`
- Only requirement: subclass `Controller` and define `compute_control()`
- Encourages user-provided implementations without modifying core code

### **Observable-Observer Pattern**:
- Environment tracks state and makes it observable via `obs()` function
- Controllers observe state through Gymnasium interface
- Callbacks allow controllers to react to episode transitions

---

## 9. Critical Execution Frequencies & Timing

| Component | Frequency | Period | Role |
|-----------|-----------|--------|------|
| Physics simulation | 500 Hz | 2 ms | Accurate drone dynamics |
| Environment step | 50 Hz | 20 ms | Control loop tick |
| Observation update | 50 Hz | 20 ms | State estimation |
| Controller computation | 50 Hz | 20 ms | Control command generation |
| Rendering | ~60 fps | ~16 ms | Visualization (async) |
| Real hardware loop | 50 Hz | 20 ms | Deployment hard deadline |

**Key: Controller must execute in < 20ms to avoid missing steps**

---

## 10. Example: Complete Control Flow for One Step

```python
# Time = 2.0 seconds, Step i = 100

# ENVIRONMENT STATE AT START OF STEP
data.steps[0] = 99  # About to become 100
data.target_gate[0] = 1  # Aiming for 2nd gate
data.sim_data.states.pos[0] ≈ [0.3, 0.2, 0.6]  # Current position

# ===== CONTROL COMPUTATION (controller side) =====
obs = {
    "pos": [0.3, 0.2, 0.6],
    "vel": [0.4, 0.1, -0.05],
    "quat": [0.0, 0.0, 0.707, 0.707],
    "ang_vel": [0.0, 0.0, 0.0],
    "target_gate": 1,
    "gates_pos": [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], ...],
    # ... more state ...
}

# Example: AstarController
action = astar_controller.compute_control(obs)  # ~5 ms
# Internally:
# 1. Check if target gate changed → no
# 2. Get current spline position: target = spline(t=2.0) ≈ [0.35, 0.23, 0.65]
# 3. Compute errors: pos_error = [0.05, 0.03, 0.05]
# 4. Run PID trackers
# 5. Return attitude command: [roll, pitch, yaw, thrust] ≈ [0.1, 0.05, -0.5, 11.5]

# ===== ENVIRONMENT STEP (environment side) =====
env.step(action)  # Calls RaceCoreEnv._step()

# 1. APPLY ACTION
# Register thrust + attitude to sim controllers
ctrl_fn = F.attitude_control(data.sim_data, action_reshaped)

# 2. PHYSICS SIMULATION (10 steps of 500 Hz)
# for _ in range(10):
#   - Integrate drone dynamics
#   - Apply control (thrust + attitude)
#   - Update state: pos, vel, quat, ang_vel
#   - Collision detection with gates/obstacles
# After 10 steps: position advances ~[0.04, 0.01, 0.0] (2 cm forward)

# 3. ENVIRONMENT LOGIC
# Check gate passage: did drone cross gate plane within bounds?
gate_passed(
    drone_pos=data.sim_data.states.pos[0],
    last_drone_pos=[0.3, 0.2, 0.6],
    gate_pos=[1.05, 0.75, 1.2],
    gate_quat=gates_quat[1],
) → False (still 0.7m away)

# Update target gate (if passage detected):
# target_gate remains 1

# Mark for reset (if collision or out of bounds):
# No collision detected → not marked

# 4. COMPUTE OBSERVATION
obs_next = {
    "pos": [0.34, 0.21, 0.60],  # Updated from physics
    "vel": [0.4, 0.1, -0.05],
    "quat": [0.0, 0.0, 0.707, 0.707],
    "ang_vel": [0.0, 0.0, 0.0],
    "target_gate": 1,  # No change
    "gates_pos": [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], ...],  # Unchanged
    # ... more state ...
}

# 5. COMPUTE REWARD
reward = 0.0  # No gate passage
# (Usually reward = 0 each step, +1.0 when passing a gate)

# 6. CHECK TERMINATION/TRUNCATION
terminated = False  # No collision
truncated = False   # steps (100) < max_episode_steps (1500)

# 7. ENVIRONMENT RETURNS
return obs_next, reward=0.0, terminated=False, truncated=False, info={}

# ===== CONTROLLER CALLBACK =====
finished = astar_controller.step_callback(
    action=[0.1, 0.05, -0.5, 11.5],
    obs=obs_next,
    reward=0.0,
    terminated=False,
    truncated=False,
    info={}
)
# Returns False (not finished)

# ===== BACK TO NEXT CONTROL ITERATION =====
i += 1  # i = 101
# Loop back to "CONTROL COMPUTATION"
```

---

## Summary

This drone racing system presents a well-architected separation between:

- **Planning** (A*, splines, MPC) → generates reference trajectories
- **Control** (PID, attitude commands) → tracks trajectories in real-time
- **Simulation** (JAX-accelerated physics) → realistic drone dynamics
- **Interface** (Gymnasium standard) → interchangeable controllers

The modular design enables rapid prototyping of new controllers while the JAX-based core allows vectorized training and deployment at scale.
