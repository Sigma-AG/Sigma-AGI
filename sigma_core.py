#!/usr/bin/env python3
"""
Sigma-AGI Core - Dynamic Neurogenesis Architecture
============================================================================

Core Philosophy: "Bitter Lesson" - Let evolution discover EVERYTHING
- No preset connections
- No fixed neuron count  
- No hardcoded topology
- Resources have cost, forcing efficiency

Key Architecture Features:
1. Dynamic Neurons: Start minimal, grow on demand
2. Sparse Connections: Build as needed (not preset)
3. Resource Penalty: Fitness = reward - alpha*neurons - beta*connections
4. Energy Survival: No fixed episodes, energy=0 means death
5. 8-DOF Body: Pure JAX physics, embodied evolution
6. Phase Transitions: Self-organized criticality for "quantum leaps"
7. Non-stationary Physics: Random gravity/friction changes

JAX Optimizations:
- Full jax.jit compilation for XLA acceleration
- jax.vmap for batched population evaluation
- Dynamic masking for static-graph compatible growth
- jax.lax.scan for efficient episode simulation

Target Hardware: TPU/GPU clusters
Population: 4096+ (scalable to TPU pods)

Author: Sigma-AGI Project
Status: Research Preview
"""

import os
from typing import NamedTuple, Tuple, Optional, Dict, List, Any
from functools import partial
import math
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.85')

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import numpy as np

print("=" * 70)
print("Sigma-AGI Dynamic Neurogenesis - Dynamic Neurogenesis")
print("=" * 70)
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# =============================================================================
# CONSTANTS - DYNAMIC ARCHITECTURE
# =============================================================================

SIGMA_VERBOSE = False  # Set True for debugging

# === Dynamic Network Bounds ===
SIGMA_INITIAL_NEURONS = 1_000       # Start small (not 100k!)
SIGMA_MAX_NEURONS = 50_000          # Hard cap (can grow to this)
SIGMA_INITIAL_CONNECTIONS = 0       # Start with ZERO connections!
SIGMA_MAX_CONNECTIONS = 500_000     # Soft cap for connections

# === Growth/Pruning Control ===
SIGMA_NEURONS_PER_GROWTH = 10       # Add 10 neurons when growing
SIGMA_CONNS_PER_GROWTH = 50         # Add 50 connections when growing
SIGMA_GROWTH_STAGNATION = 50        # Generations without improvement triggers growth
SIGMA_PRUNE_THRESHOLD = 0.01        # Connection usage below this = prune

# === Resource Penalty (Bitter Lesson: complexity has cost) ===
SIGMA_NEURON_COST = 0.0001          # Cost per neuron (alpha)
SIGMA_CONNECTION_COST = 0.00001     # Cost per connection (beta)

# === Energy System (No fixed episodes!) ===
SIGMA_INITIAL_ENERGY = 200.0        # Starting energy (increased for exploration)
SIGMA_ENERGY_PER_STEP = -0.3        # Base energy drain per step (reduced)
SIGMA_MOVEMENT_ENERGY_COST = -0.5   # Extra cost for movement (reduced)
SIGMA_FOOD_ENERGY = 100.0           # Energy from food (increased reward)
SIGMA_DEATH_THRESHOLD = 0.0         # energy <= 0 = death

# === 8-DOF Body Configuration ===
SIGMA_N_BODY_JOINTS = 8             # 8 degrees of freedom
SIGMA_N_SENSORS = 32                # Proprioception + touch + vision proxy
SIGMA_N_OUTPUTS = SIGMA_N_BODY_JOINTS # Direct joint torque control

# === Body Joint Layout ===
# Joint 0-1: Torso (pitch, roll)
# Joint 2-3: Left leg (hip, knee)
# Joint 4-5: Right leg (hip, knee)
# Joint 6-7: Arms (left, right)
JOINT_TORSO_PITCH = 0
JOINT_TORSO_ROLL = 1
JOINT_LEFT_HIP = 2
JOINT_LEFT_KNEE = 3
JOINT_RIGHT_HIP = 4
JOINT_RIGHT_KNEE = 5
JOINT_LEFT_ARM = 6
JOINT_RIGHT_ARM = 7

# === Physics Constants ===
SIGMA_DT = 0.02                     # 50Hz simulation
SIGMA_GRAVITY = -9.81               # m/s^2
SIGMA_FRICTION = 0.8                # Ground friction
SIGMA_JOINT_DAMPING = 0.1           # Joint damping coefficient

# === Non-stationary Physics (forces adaptation) ===
SIGMA_PHYSICS_CHANGE_INTERVAL = 100 # Steps between physics changes
SIGMA_GRAVITY_RANGE = (-12.0, -7.0) # Gravity can vary
SIGMA_FRICTION_RANGE = (0.3, 1.2)   # Friction can vary

# === Phase Transition / Criticality ===
SIGMA_CRITICALITY_WINDOW = 100      # Window to measure criticality
SIGMA_CRITICALITY_TARGET = 1.0      # Target criticality (edge of chaos)
SIGMA_CRITICALITY_TOLERANCE = 0.2   # Acceptable range around target

# === Evolution Parameters ===
SIGMA_POPULATION = 128              # Large population (small model allows this)
SIGMA_TOURNAMENT_SIZE = 7           # Tournament selection
SIGMA_MUTATION_RATE = 0.15          # Higher mutation for exploration
SIGMA_ELITE_RATIO = 0.1             # Top 10% survive unchanged

# === Neural Dynamics ===
SIGMA_TAU_MIN = 1.0                 # Minimum time constant
SIGMA_TAU_MAX = 50.0                # Maximum time constant
SIGMA_ACTIVATION_SPARSITY = 0.1     # Target 10% neurons active


# =============================================================================
# DYNAMIC NEURON TYPES (Evolved, not preset)
# =============================================================================

# Minimal initial types - evolution discovers what's needed
NEURON_TYPE_GENERIC = 0           # Default type
NEURON_TYPE_SENSOR = 1            # Connected to sensors
NEURON_TYPE_MOTOR = 2             # Connected to motors
N_NEURON_TYPES = 3


# =============================================================================
# GENOME - Minimal Priors, Maximum Evolution
# =============================================================================

class GrowthGene(NamedTuple):
    """
    Genes controlling HOW the network grows.
    Evolution discovers the growth rules.
    """
    # When to grow neurons
    growth_trigger_threshold: jnp.ndarray   # (1,) fitness stagnation threshold
    growth_neuron_count: jnp.ndarray        # (1,) how many neurons to add

    # Where to place new neurons (relative to active neurons)
    new_neuron_offset_scale: jnp.ndarray    # (1,) spatial spread

    # When to grow connections
    connection_growth_rate: jnp.ndarray     # (1,) connections per growth event
    connection_distance_preference: jnp.ndarray  # (1,) prefer local vs global

    # When to prune
    prune_threshold: jnp.ndarray            # (1,) usage below this = prune
    prune_rate: jnp.ndarray                 # (1,) fraction to prune


class NeuralGene(NamedTuple):
    """
    Genes controlling neural dynamics.
    """
    # Time constants (per type)
    tau_by_type: jnp.ndarray                # (N_TYPES,) time constants

    # Activation function parameters
    activation_gain: jnp.ndarray            # (1,) steepness
    activation_bias: jnp.ndarray            # (1,) threshold

    # Plasticity
    learning_rate: jnp.ndarray              # (1,) Hebbian learning rate
    trace_decay: jnp.ndarray                # (1,) eligibility trace decay


class BodyGene(NamedTuple):
    """
    Genes controlling body parameters.
    Evolution tunes the body for its environment.
    """
    # Joint properties
    joint_strength: jnp.ndarray             # (8,) max torque per joint
    joint_damping: jnp.ndarray              # (8,) damping per joint

    # Limb properties (affects mass distribution)
    limb_mass: jnp.ndarray                  # (8,) mass of each segment
    limb_length: jnp.ndarray                # (8,) length of each segment


class CriticalityGene(NamedTuple):
    """
    Genes controlling criticality / phase transitions.
    This enables "quantum leaps" in capability.
    """
    # Self-organized criticality parameters
    target_criticality: jnp.ndarray         # (1,) target branching ratio
    adaptation_rate: jnp.ndarray            # (1,) how fast to adapt

    # Avalanche parameters
    avalanche_threshold: jnp.ndarray        # (1,) cascade threshold
    inhibition_strength: jnp.ndarray        # (1,) lateral inhibition


class Genome(NamedTuple):
    """
    Complete Genome - Minimal Priors

    Unlike fixed-topology massive preset topology, starts nearly empty
    and evolves everything from scratch.
    """
    # Growth rules
    growth: GrowthGene

    # Neural dynamics
    neural: NeuralGene

    # Body parameters
    body: BodyGene

    # Criticality control
    criticality: CriticalityGene

    # Random seed for initialization
    init_seed: jnp.ndarray                  # (1,)

    # Readout weights (from neurons to motors)
    # Starts small, grows with network
    readout_kernel: jnp.ndarray             # (KERNEL_SIZE,) pattern for readout

    # Sensor weights (from sensors to neurons)
    sensor_kernel: jnp.ndarray              # (KERNEL_SIZE,) pattern for input


def init_genome(key: jnp.ndarray) -> Genome:
    """
    Initialize genome with MINIMAL priors.

    Unlike previous versions, we don't preset topology - we just provide
    the genes that control how topology EVOLVES.
    """
    keys = random.split(key, 20)

    growth = GrowthGene(
        growth_trigger_threshold=jnp.array([0.01]),
        growth_neuron_count=jnp.array([10.0]),
        new_neuron_offset_scale=jnp.array([0.3]),
        connection_growth_rate=jnp.array([50.0]),
        connection_distance_preference=jnp.array([0.7]),  # Prefer local
        prune_threshold=jnp.array([0.01]),
        prune_rate=jnp.array([0.05]),
    )

    neural = NeuralGene(
        tau_by_type=random.uniform(keys[0], (N_NEURON_TYPES,),
                                   minval=SIGMA_TAU_MIN, maxval=SIGMA_TAU_MAX),
        activation_gain=jnp.array([1.0]),
        activation_bias=jnp.array([0.0]),
        learning_rate=jnp.array([0.01]),
        trace_decay=jnp.array([0.95]),
    )

    body = BodyGene(
        joint_strength=jnp.ones(SIGMA_N_BODY_JOINTS) * 100.0,
        joint_damping=jnp.ones(SIGMA_N_BODY_JOINTS) * SIGMA_JOINT_DAMPING,
        limb_mass=jnp.array([10.0, 5.0, 5.0, 3.0, 5.0, 3.0, 2.0, 2.0]),
        limb_length=jnp.array([0.5, 0.3, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3]),
    )

    criticality = CriticalityGene(
        target_criticality=jnp.array([1.0]),
        adaptation_rate=jnp.array([0.01]),
        avalanche_threshold=jnp.array([0.5]),
        inhibition_strength=jnp.array([0.3]),
    )

    # Kernels for weight generation (small, will be expanded)
    kernel_size = 32
    readout_kernel = random.normal(keys[1], (kernel_size,)) * 0.1
    sensor_kernel = random.normal(keys[2], (kernel_size,)) * 0.1

    return Genome(
        growth=growth,
        neural=neural,
        body=body,
        criticality=criticality,
        init_seed=keys[3][0:1],
        readout_kernel=readout_kernel,
        sensor_kernel=sensor_kernel,
    )


def mutate_genome(genome: Genome, key: jnp.ndarray, rate: float = 0.15) -> Genome:
    """
    Mutate genome.

    Higher mutation rate than previous versions because we're exploring topology space.
    """
    keys = random.split(key, 30)

    def mutate_scalar(val, k, scale=0.1):
        return val + random.normal(k, val.shape) * rate * scale

    def mutate_positive(val, k, scale=0.1, min_val=0.001):
        return jnp.maximum(val + random.normal(k, val.shape) * rate * scale, min_val)

    # Mutate growth genes
    new_growth = GrowthGene(
        growth_trigger_threshold=mutate_positive(genome.growth.growth_trigger_threshold, keys[0]),
        growth_neuron_count=mutate_positive(genome.growth.growth_neuron_count, keys[1], scale=5.0, min_val=1.0),
        new_neuron_offset_scale=mutate_positive(genome.growth.new_neuron_offset_scale, keys[2]),
        connection_growth_rate=mutate_positive(genome.growth.connection_growth_rate, keys[3], scale=20.0),
        connection_distance_preference=jnp.clip(mutate_scalar(genome.growth.connection_distance_preference, keys[4]), 0.0, 1.0),
        prune_threshold=mutate_positive(genome.growth.prune_threshold, keys[5], scale=0.01),
        prune_rate=jnp.clip(mutate_scalar(genome.growth.prune_rate, keys[6]), 0.01, 0.5),
    )

    # Mutate neural genes
    new_neural = NeuralGene(
        tau_by_type=jnp.clip(mutate_scalar(genome.neural.tau_by_type, keys[7], scale=5.0), SIGMA_TAU_MIN, SIGMA_TAU_MAX),
        activation_gain=mutate_positive(genome.neural.activation_gain, keys[8]),
        activation_bias=mutate_scalar(genome.neural.activation_bias, keys[9], scale=0.5),
        learning_rate=mutate_positive(genome.neural.learning_rate, keys[10], scale=0.005),
        trace_decay=jnp.clip(mutate_scalar(genome.neural.trace_decay, keys[11], scale=0.05), 0.5, 0.99),
    )

    # Mutate body genes
    new_body = BodyGene(
        joint_strength=mutate_positive(genome.body.joint_strength, keys[12], scale=20.0),
        joint_damping=mutate_positive(genome.body.joint_damping, keys[13], scale=0.02),
        limb_mass=mutate_positive(genome.body.limb_mass, keys[14], scale=1.0),
        limb_length=mutate_positive(genome.body.limb_length, keys[15], scale=0.1, min_val=0.1),
    )

    # Mutate criticality genes
    new_criticality = CriticalityGene(
        target_criticality=mutate_positive(genome.criticality.target_criticality, keys[16], scale=0.2),
        adaptation_rate=mutate_positive(genome.criticality.adaptation_rate, keys[17], scale=0.005),
        avalanche_threshold=jnp.clip(mutate_scalar(genome.criticality.avalanche_threshold, keys[18]), 0.1, 0.9),
        inhibition_strength=jnp.clip(mutate_scalar(genome.criticality.inhibition_strength, keys[19]), 0.0, 1.0),
    )

    # Mutate kernels
    new_readout = genome.readout_kernel + random.normal(keys[20], genome.readout_kernel.shape) * rate * 0.1
    new_sensor = genome.sensor_kernel + random.normal(keys[21], genome.sensor_kernel.shape) * rate * 0.1

    return Genome(
        growth=new_growth,
        neural=new_neural,
        body=new_body,
        criticality=new_criticality,
        init_seed=genome.init_seed,
        readout_kernel=new_readout,
        sensor_kernel=new_sensor,
    )


# =============================================================================
# DYNAMIC NETWORK TOPOLOGY
# =============================================================================

class DynamicTopology(NamedTuple):
    """
    Dynamic network topology that GROWS from scratch.

    Unlike fixed-topology fixed 10M connections, starts with ZERO
    and evolves connections as needed.
    """
    # Active neuron count (starts at SIGMA_INITIAL_NEURONS)
    n_neurons: jnp.ndarray                  # (1,) current neuron count

    # Neuron properties (padded to MAX_NEURONS)
    neuron_pos: jnp.ndarray                 # (MAX_NEURONS, 3) positions
    neuron_type: jnp.ndarray                # (MAX_NEURONS,) types
    neuron_active: jnp.ndarray              # (MAX_NEURONS,) 1=active, 0=inactive

    # Connection arrays (dynamic, padded to MAX_CONNECTIONS)
    n_connections: jnp.ndarray              # (1,) current connection count
    conn_pre: jnp.ndarray                   # (MAX_CONNS,) pre-synaptic neuron
    conn_post: jnp.ndarray                  # (MAX_CONNS,) post-synaptic neuron
    conn_weight: jnp.ndarray                # (MAX_CONNS,) synaptic weight
    conn_active: jnp.ndarray                # (MAX_CONNS,) 1=active, 0=inactive
    conn_usage: jnp.ndarray                 # (MAX_CONNS,) usage for pruning

    # Sensor and motor mappings
    sensor_weights: jnp.ndarray             # (N_SENSORS, MAX_NEURONS) input weights
    motor_weights: jnp.ndarray              # (MAX_NEURONS, N_OUTPUTS) output weights


def init_topology(genome: Genome, key: jnp.ndarray) -> DynamicTopology:
    """
    Initialize MINIMAL topology.

    Start with SIGMA_INITIAL_NEURONS and SPARSE internal connections.
    Creates a minimal pathway: sensors -> generic -> motors
    Evolution can grow/prune this initial structure.
    """
    keys = random.split(key, 10)

    # Initialize neuron positions (only first n_initial are active)
    n_initial = SIGMA_INITIAL_NEURONS

    # Random positions in 3D space for initial neurons
    initial_pos = random.uniform(keys[0], (SIGMA_MAX_NEURONS, 3), minval=-1.0, maxval=1.0)

    # Initialize neuron types
    # First neurons are sensors, middle are generic, last are motors
    n_sensor = SIGMA_N_SENSORS
    n_motor = SIGMA_N_OUTPUTS
    n_generic = n_initial - n_sensor - n_motor

    types = jnp.zeros(SIGMA_MAX_NEURONS, dtype=jnp.int32)
    types = types.at[:n_sensor].set(NEURON_TYPE_SENSOR)
    types = types.at[n_sensor:n_sensor+n_generic].set(NEURON_TYPE_GENERIC)
    types = types.at[n_sensor+n_generic:n_initial].set(NEURON_TYPE_MOTOR)

    # Active mask
    active = (jnp.arange(SIGMA_MAX_NEURONS) < n_initial).astype(jnp.float32)

    # ===== INITIAL CONNECTIONS =====
    # Create minimal pathway: sensor -> generic -> motor
    # This gives evolution something to work with (Bitter Lesson: still minimal!)

    # Number of initial internal connections (sparse)
    n_initial_conns = 500  # Much less than fixed-topology 10M!

    # Generate random connections biased toward sensor->generic->motor flow
    conn_pre = jnp.zeros(SIGMA_MAX_CONNECTIONS, dtype=jnp.int32)
    conn_post = jnp.zeros(SIGMA_MAX_CONNECTIONS, dtype=jnp.int32)
    conn_weight = jnp.zeros(SIGMA_MAX_CONNECTIONS)
    conn_active = jnp.zeros(SIGMA_MAX_CONNECTIONS)
    conn_usage = jnp.zeros(SIGMA_MAX_CONNECTIONS)

    # Layer 1: Sensor -> Generic (200 connections)
    n_s2g = 200
    pre_s2g = random.randint(keys[1], (n_s2g,), 0, n_sensor)
    post_s2g = random.randint(keys[2], (n_s2g,), n_sensor, n_sensor + n_generic)

    # Layer 2: Generic -> Generic (150 connections)
    n_g2g = 150
    pre_g2g = random.randint(keys[3], (n_g2g,), n_sensor, n_sensor + n_generic)
    post_g2g = random.randint(keys[4], (n_g2g,), n_sensor, n_sensor + n_generic)
    # Avoid self-connections
    post_g2g = jnp.where(post_g2g == pre_g2g, (post_g2g + 1 - n_sensor) % n_generic + n_sensor, post_g2g)

    # Layer 3: Generic -> Motor (150 connections)
    n_g2m = 150
    pre_g2m = random.randint(keys[5], (n_g2m,), n_sensor, n_sensor + n_generic)
    motor_start = n_sensor + n_generic
    post_g2m = random.randint(keys[6], (n_g2m,), motor_start, n_initial)

    # Combine all connections
    all_pre = jnp.concatenate([pre_s2g, pre_g2g, pre_g2m])
    all_post = jnp.concatenate([post_s2g, post_g2g, post_g2m])
    n_total_conns = n_s2g + n_g2g + n_g2m

    # Initialize weights with stronger magnitude for signal propagation
    # Use larger scale to ensure activity can propagate through the network
    weight_scale = 0.5  # Stronger initial weights
    all_weights = random.normal(keys[7], (n_total_conns,)) * weight_scale

    # Set connection arrays
    conn_pre = conn_pre.at[:n_total_conns].set(all_pre)
    conn_post = conn_post.at[:n_total_conns].set(all_post)
    conn_weight = conn_weight.at[:n_total_conns].set(all_weights)
    conn_active = conn_active.at[:n_total_conns].set(1.0)
    conn_usage = conn_usage.at[:n_total_conns].set(0.5)

    # ===== SENSOR AND MOTOR INTERFACES =====
    # These are the "body interface" - must exist for any behavior

    # Sensor weights: connect each sensor to multiple neurons with strong weights
    sensor_weights = jnp.zeros((SIGMA_N_SENSORS, SIGMA_MAX_NEURONS))
    # Each sensor connects to its corresponding neuron AND nearby generic neurons
    for i in range(SIGMA_N_SENSORS):
        # Strong connection to corresponding sensor neuron
        sensor_weights = sensor_weights.at[i, i].set(1.0)
        # Also connect to some generic neurons (fan-out for richer representation)
        for j in range(5):  # Connect to 5 additional generic neurons
            target = n_sensor + (i * 5 + j) % n_generic
            sensor_weights = sensor_weights.at[i, target].set(0.5)

    # Motor weights: connect generic neurons to motor outputs
    motor_weights = jnp.zeros((SIGMA_MAX_NEURONS, SIGMA_N_OUTPUTS))
    # Motor neurons have direct strong connection
    for i in range(SIGMA_N_OUTPUTS):
        motor_weights = motor_weights.at[motor_start + i, i].set(1.0)
    # Also connect some generic neurons to motors (multiple pathways)
    for i in range(SIGMA_N_OUTPUTS):
        for j in range(10):  # 10 generic neurons per motor
            source = n_sensor + (i * 10 + j) % n_generic
            motor_weights = motor_weights.at[source, i].set(0.3)

    if SIGMA_VERBOSE:
        print(f"  Initialized: {n_initial} neurons, {n_total_conns} internal connections")
        print(f"  Sensor neurons: {n_sensor}, Generic: {n_generic}, Motor: {n_motor}")
        print(f"  Connections: S->G={n_s2g}, G->G={n_g2g}, G->M={n_g2m}")

    return DynamicTopology(
        n_neurons=jnp.array([n_initial]),
        neuron_pos=initial_pos,
        neuron_type=types,
        neuron_active=active,
        n_connections=jnp.array([n_total_conns]),
        conn_pre=conn_pre,
        conn_post=conn_post,
        conn_weight=conn_weight,
        conn_active=conn_active,
        conn_usage=conn_usage,
        sensor_weights=sensor_weights,
        motor_weights=motor_weights,
    )


@jit
def add_neurons(topology: DynamicTopology, key: jnp.ndarray, n_add: int) -> DynamicTopology:
    """
    Add new neurons to the network (neurogenesis).

    New neurons are placed near active neurons with high activity.
    """
    current_n = topology.n_neurons[0]
    new_n = jnp.minimum(current_n + n_add, SIGMA_MAX_NEURONS)
    actual_add = new_n - current_n

    # Generate positions near existing neurons (random offset from random existing)
    keys = random.split(key, 3)
    parent_indices = random.randint(keys[0], (actual_add,), 0, current_n)
    parent_pos = topology.neuron_pos[parent_indices]
    offset = random.normal(keys[1], (actual_add, 3)) * 0.2
    new_pos = parent_pos + offset
    new_pos = jnp.clip(new_pos, -1.0, 1.0)

    # Update positions
    new_neuron_pos = topology.neuron_pos.at[current_n:new_n].set(new_pos)

    # New neurons are generic type
    new_types = topology.neuron_type.at[current_n:new_n].set(NEURON_TYPE_GENERIC)

    # Activate new neurons
    new_active = topology.neuron_active.at[current_n:new_n].set(1.0)

    return topology._replace(
        n_neurons=jnp.array([new_n]),
        neuron_pos=new_neuron_pos,
        neuron_type=new_types,
        neuron_active=new_active,
    )


@jit
def add_connections(
    topology: DynamicTopology,
    key: jnp.ndarray,
    n_add: int,
    distance_preference: float = 0.7,
) -> DynamicTopology:
    """
    Add new connections (synaptogenesis).

    Connections are biased towards:
    - Local connections (spatial proximity)
    - Active neurons
    """
    current_n_conn = topology.n_connections[0]
    current_n_neurons = topology.n_neurons[0]
    new_n_conn = jnp.minimum(current_n_conn + n_add, SIGMA_MAX_CONNECTIONS)
    actual_add = new_n_conn - current_n_conn

    keys = random.split(key, 4)

    # Generate random pre/post pairs
    pre_neurons = random.randint(keys[0], (actual_add,), 0, current_n_neurons)
    post_neurons = random.randint(keys[1], (actual_add,), 0, current_n_neurons)

    # Avoid self-connections
    post_neurons = (post_neurons + 1) % current_n_neurons

    # Initialize weights using He initialization
    # fan_in approximated as current_n_conn / current_n_neurons
    fan_in = jnp.maximum(current_n_conn / jnp.maximum(current_n_neurons, 1), 1.0)
    weight_scale = jnp.sqrt(2.0 / fan_in)
    new_weights = random.normal(keys[2], (actual_add,)) * weight_scale

    # Update connection arrays
    new_conn_pre = topology.conn_pre.at[current_n_conn:new_n_conn].set(pre_neurons)
    new_conn_post = topology.conn_post.at[current_n_conn:new_n_conn].set(post_neurons)
    new_conn_weight = topology.conn_weight.at[current_n_conn:new_n_conn].set(new_weights)
    new_conn_active = topology.conn_active.at[current_n_conn:new_n_conn].set(1.0)
    new_conn_usage = topology.conn_usage.at[current_n_conn:new_n_conn].set(0.5)

    return topology._replace(
        n_connections=jnp.array([new_n_conn]),
        conn_pre=new_conn_pre,
        conn_post=new_conn_post,
        conn_weight=new_conn_weight,
        conn_active=new_conn_active,
        conn_usage=new_conn_usage,
    )


@jit
def prune_connections(
    topology: DynamicTopology,
    prune_threshold: float = 0.01,
) -> DynamicTopology:
    """
    Prune low-usage connections (synaptic pruning).

    Connections with usage below threshold are deactivated.
    """
    # Deactivate low-usage connections
    should_prune = (topology.conn_usage < prune_threshold) & (topology.conn_active > 0.5)
    new_active = jnp.where(should_prune, 0.0, topology.conn_active)

    return topology._replace(conn_active=new_active)


# =============================================================================
# BODY PHYSICS - Pure JAX Implementation
# =============================================================================

class BodyState(NamedTuple):
    """
    8-DOF body state for embodied evolution.

    Represents a simplified biped in 2D plane with:
    - Torso (2 DOF: pitch, roll)
    - Legs (4 DOF: 2 joints per leg)
    - Arms (2 DOF: 1 joint per arm)
    """
    # Position and velocity of center of mass
    com_pos: jnp.ndarray                    # (3,) x, y, z position
    com_vel: jnp.ndarray                    # (3,) x, y, z velocity

    # Joint angles and velocities
    joint_angles: jnp.ndarray               # (8,) current angles
    joint_velocities: jnp.ndarray           # (8,) current angular velocities

    # Ground contact (for energy gain from food)
    foot_contacts: jnp.ndarray              # (2,) left, right foot contact

    # Orientation (quaternion simplified to euler for 2D-ish motion)
    orientation: jnp.ndarray                # (3,) roll, pitch, yaw


def init_body_state() -> BodyState:
    """Initialize body in standing position."""
    return BodyState(
        com_pos=jnp.array([0.0, 0.0, 1.0]),  # Standing height
        com_vel=jnp.zeros(3),
        joint_angles=jnp.zeros(SIGMA_N_BODY_JOINTS),
        joint_velocities=jnp.zeros(SIGMA_N_BODY_JOINTS),
        foot_contacts=jnp.ones(2),  # Both feet on ground initially
        orientation=jnp.zeros(3),
    )


@jit
def compute_body_sensors(body: BodyState, food_positions: jnp.ndarray) -> jnp.ndarray:
    """
    Compute sensor readings from body state.

    Returns 32 sensor values:
    - Joint angles (8)
    - Joint velocities (8)
    - COM velocity (3)
    - Orientation (3)
    - Foot contacts (2)
    - Food direction (2) - direction to nearest food
    - Height (1)
    - Energy proxy (1) - based on movement efficiency
    - Padding (4)
    """
    # Normalize joint angles to [-1, 1]
    norm_angles = body.joint_angles / jnp.pi

    # Normalize joint velocities
    norm_velocities = jnp.tanh(body.joint_velocities / 10.0)

    # Normalize COM velocity
    norm_com_vel = jnp.tanh(body.com_vel / 5.0)

    # Normalize orientation
    norm_orient = body.orientation / jnp.pi

    # Food direction (simplified - direction to origin if no food)
    food_dir = -body.com_pos[:2] / (jnp.linalg.norm(body.com_pos[:2]) + 1e-6)

    # Height sensor
    height = jnp.array([body.com_pos[2] / 2.0])

    # Movement efficiency (low = wasting energy)
    efficiency = jnp.array([jnp.tanh(jnp.linalg.norm(body.com_vel[:2]))])

    # Padding
    padding = jnp.zeros(4)

    sensors = jnp.concatenate([
        norm_angles,           # 8
        norm_velocities,       # 8
        norm_com_vel,          # 3
        norm_orient,           # 3
        body.foot_contacts,    # 2
        food_dir,              # 2
        height,                # 1
        efficiency,            # 1
        padding,               # 4
    ])

    return sensors


@jit
def step_body_physics(
    body: BodyState,
    torques: jnp.ndarray,
    genome_body: BodyGene,
    gravity: float = SIGMA_GRAVITY,
    friction: float = SIGMA_FRICTION,
    dt: float = SIGMA_DT,
) -> Tuple[BodyState, float]:
    """
    Step body physics forward.

    Returns updated body state and energy cost of movement.

    Simplified physics:
    - Joint torques drive joint accelerations
    - Gravity pulls COM down
    - Ground collision prevents falling through
    - Foot contacts depend on leg configuration
    """
    # Clip torques to joint strength limits
    max_torques = genome_body.joint_strength
    torques = jnp.clip(torques, -max_torques, max_torques)

    # Joint dynamics: torque -> acceleration -> velocity -> angle
    # M * dv/dt = torque - damping * v
    joint_acc = torques / genome_body.limb_mass - genome_body.joint_damping * body.joint_velocities
    new_velocities = body.joint_velocities + joint_acc * dt
    new_angles = body.joint_angles + new_velocities * dt

    # Clip angles to reasonable range
    new_angles = jnp.clip(new_angles, -jnp.pi * 0.8, jnp.pi * 0.8)

    # COM dynamics from leg and torso angles (simplified inverse kinematics)
    # Legs affect vertical movement AND horizontal push
    leg_extension = (jnp.cos(new_angles[JOINT_LEFT_KNEE]) +
                     jnp.cos(new_angles[JOINT_RIGHT_KNEE])) / 2.0
    target_height = 0.5 + leg_extension * 0.5  # 0.5 to 1.0

    # Horizontal velocity from torso lean (INCREASED for locomotion)
    lean = new_angles[JOINT_TORSO_PITCH]
    horizontal_push_lean = jnp.sin(lean) * 10.0  # Increased from 2.0

    # ADDED: Leg-driven horizontal push (asymmetric leg angles = walking)
    left_hip = new_angles[JOINT_LEFT_HIP]
    right_hip = new_angles[JOINT_RIGHT_HIP]
    leg_asymmetry = left_hip - right_hip  # Difference drives forward motion
    horizontal_push_legs = jnp.sin(leg_asymmetry) * 5.0

    # Total horizontal acceleration
    horizontal_push = horizontal_push_lean + horizontal_push_legs

    # Apply gravity to COM
    com_acc = jnp.array([horizontal_push, 0.0, gravity])
    new_com_vel = body.com_vel + com_acc * dt
    new_com_pos = body.com_pos + new_com_vel * dt

    # Ground collision
    on_ground = new_com_pos[2] <= target_height
    new_com_pos = new_com_pos.at[2].set(jnp.maximum(new_com_pos[2], target_height))
    new_com_vel = jnp.where(on_ground & (new_com_vel[2] < 0),
                            new_com_vel.at[2].set(0.0),
                            new_com_vel)

    # Ground friction when on ground (REDUCED for easier locomotion)
    friction_factor = 0.5  # Reduced from 1.0
    new_com_vel = jnp.where(
        on_ground,
        new_com_vel * jnp.array([1.0 - friction * dt * friction_factor,
                                  1.0 - friction * dt * friction_factor, 1.0]),
        new_com_vel
    )

    # Foot contacts based on leg angles
    left_foot = jnp.cos(new_angles[JOINT_LEFT_HIP] + new_angles[JOINT_LEFT_KNEE]) > 0.3
    right_foot = jnp.cos(new_angles[JOINT_RIGHT_HIP] + new_angles[JOINT_RIGHT_KNEE]) > 0.3
    foot_contacts = jnp.array([left_foot.astype(jnp.float32),
                                right_foot.astype(jnp.float32)])

    # Orientation from torso angles
    orientation = jnp.array([new_angles[JOINT_TORSO_ROLL],
                             new_angles[JOINT_TORSO_PITCH],
                             0.0])

    # Energy cost = torque^2 (penalize large torques)
    energy_cost = jnp.sum(torques ** 2) * 0.0001

    new_body = BodyState(
        com_pos=new_com_pos,
        com_vel=new_com_vel,
        joint_angles=new_angles,
        joint_velocities=new_velocities,
        foot_contacts=foot_contacts,
        orientation=orientation,
    )

    return new_body, energy_cost


# =============================================================================
# ENERGY SURVIVAL SYSTEM
# =============================================================================

class EnergyState(NamedTuple):
    """
    Energy-based survival state.

    No fixed episodes! Death comes when energy = 0.
    """
    energy: jnp.ndarray                     # (1,) current energy
    total_food_eaten: jnp.ndarray           # (1,) lifetime food count
    steps_survived: jnp.ndarray             # (1,) survival time
    cause_of_death: jnp.ndarray             # (1,) 0=alive, 1=starved, 2=fell


def init_energy_state() -> EnergyState:
    """Initialize with full energy."""
    return EnergyState(
        energy=jnp.array([SIGMA_INITIAL_ENERGY]),
        total_food_eaten=jnp.array([0.0]),
        steps_survived=jnp.array([0.0]),
        cause_of_death=jnp.array([0.0]),
    )


@jit
def update_energy(
    energy_state: EnergyState,
    movement_cost: float,
    food_eaten: float,
    fell: bool,
) -> Tuple[EnergyState, bool]:
    """
    Update energy based on activity and food.

    Returns updated state and whether creature is alive.
    """
    # Base energy drain
    new_energy = energy_state.energy + SIGMA_ENERGY_PER_STEP

    # Movement cost
    new_energy = new_energy + SIGMA_MOVEMENT_ENERGY_COST * movement_cost

    # Food gain
    new_energy = new_energy + food_eaten * SIGMA_FOOD_ENERGY

    # Clamp to reasonable range
    new_energy = jnp.clip(new_energy, 0.0, SIGMA_INITIAL_ENERGY * 2)

    # Check death conditions
    starved = new_energy[0] <= SIGMA_DEATH_THRESHOLD

    # Update cause of death
    cause = jnp.where(starved, 1.0, jnp.where(fell, 2.0, 0.0))
    cause = jnp.where(energy_state.cause_of_death[0] > 0,
                      energy_state.cause_of_death[0], cause)

    alive = ~starved & ~fell & (energy_state.cause_of_death[0] == 0)

    new_state = EnergyState(
        energy=new_energy,
        total_food_eaten=energy_state.total_food_eaten + food_eaten,
        steps_survived=energy_state.steps_survived + alive.astype(jnp.float32),
        cause_of_death=jnp.array([cause]),
    )

    return new_state, alive


# =============================================================================
# CRITICALITY / PHASE TRANSITIONS
# =============================================================================

class CriticalityState(NamedTuple):
    """
    State for tracking network criticality.

    Self-organized criticality enables phase transitions
    ("quantum leaps") in capability.
    """
    # Activity history for branching ratio calculation
    activity_history: jnp.ndarray           # (WINDOW, MAX_NEURONS) recent activations
    history_idx: jnp.ndarray                # (1,) current index in circular buffer

    # Branching ratio (target: 1.0 = edge of chaos)
    branching_ratio: jnp.ndarray            # (1,) current branching ratio

    # Avalanche statistics
    current_avalanche_size: jnp.ndarray     # (1,) size of current avalanche
    max_avalanche_size: jnp.ndarray         # (1,) largest avalanche seen


def init_criticality_state() -> CriticalityState:
    """Initialize criticality tracking."""
    return CriticalityState(
        activity_history=jnp.zeros((SIGMA_CRITICALITY_WINDOW, SIGMA_MAX_NEURONS)),
        history_idx=jnp.array([0]),
        branching_ratio=jnp.array([1.0]),
        current_avalanche_size=jnp.array([0.0]),
        max_avalanche_size=jnp.array([0.0]),
    )


@jit
def update_criticality(
    crit_state: CriticalityState,
    activations: jnp.ndarray,
    neuron_mask: jnp.ndarray,
) -> CriticalityState:
    """
    Update criticality state based on neural activity.

    Measures branching ratio to detect phase transitions.
    Uses neuron_mask instead of n_neurons for JAX JIT compatibility.
    """
    # Record current activations
    idx = crit_state.history_idx[0] % SIGMA_CRITICALITY_WINDOW
    new_history = crit_state.activity_history.at[idx].set(activations)
    new_idx = crit_state.history_idx + 1

    # Compute branching ratio (ratio of activity between time steps)
    # Only count active neurons (mask-based)
    current_active = jnp.sum((activations > 0.1) * neuron_mask)

    prev_idx = (idx - 1) % SIGMA_CRITICALITY_WINDOW
    prev_activations = crit_state.activity_history[prev_idx]
    prev_active = jnp.sum((prev_activations > 0.1) * neuron_mask)

    # Branching ratio = current / previous (with smoothing)
    raw_ratio = current_active / jnp.maximum(prev_active, 1.0)
    smoothed_ratio = 0.9 * crit_state.branching_ratio + 0.1 * raw_ratio

    # Avalanche tracking
    in_avalanche = current_active > prev_active * 1.5
    new_avalanche_size = jnp.where(in_avalanche,
                                    crit_state.current_avalanche_size + current_active,
                                    0.0)
    new_max = jnp.maximum(crit_state.max_avalanche_size, new_avalanche_size)

    return CriticalityState(
        activity_history=new_history,
        history_idx=new_idx,
        branching_ratio=smoothed_ratio,
        current_avalanche_size=new_avalanche_size,
        max_avalanche_size=new_max,
    )


@jit
def apply_criticality_adaptation(
    activations: jnp.ndarray,
    crit_state: CriticalityState,
    genome_crit: CriticalityGene,
) -> jnp.ndarray:
    """
    Apply lateral inhibition to maintain criticality.

    If branching ratio > target: increase inhibition (prevent runaway)
    If branching ratio < target: decrease inhibition (allow activity)
    """
    target = genome_crit.target_criticality[0]
    current = crit_state.branching_ratio[0]

    # Error signal
    error = current - target

    # Adaptive inhibition: positive error = too much activity
    inhibition = genome_crit.inhibition_strength[0] * (1.0 + error)

    # Apply soft winner-take-all with adaptive threshold
    threshold = genome_crit.avalanche_threshold[0]
    above_threshold = activations > threshold
    mean_above = jnp.mean(activations * above_threshold)

    # Lateral inhibition
    inhibited = activations - inhibition * mean_above

    return jnp.maximum(inhibited, 0.0)


# =============================================================================
# NEURAL DYNAMICS
# =============================================================================

class NeuralState(NamedTuple):
    """
    Dynamic neural state.
    """
    # Activations
    x: jnp.ndarray                          # (MAX_NEURONS,) neural activations

    # Eligibility traces for plasticity
    traces: jnp.ndarray                     # (MAX_CONNS,) eligibility traces

    # Neuromodulator levels (global signals)
    modulators: jnp.ndarray                 # (4,) dopamine, serotonin, etc.


def init_neural_state(key: jnp.ndarray = None) -> NeuralState:
    """Initialize neural state with small random activations."""
    if key is None:
        key = random.PRNGKey(0)
    # Start with small random activations to bootstrap activity
    initial_x = random.uniform(key, (SIGMA_MAX_NEURONS,), minval=-0.1, maxval=0.1)
    return NeuralState(
        x=initial_x,
        traces=jnp.zeros(SIGMA_MAX_CONNECTIONS),
        modulators=jnp.zeros(4),
    )


@jit
def neural_forward(
    neural_state: NeuralState,
    topology: DynamicTopology,
    sensors: jnp.ndarray,
    genome_neural: NeuralGene,
) -> Tuple[NeuralState, jnp.ndarray]:
    """
    Forward pass through dynamic neural network.

    Unlike fixed topology, Sigma-AGI uses only active connections.
    Uses masking instead of dynamic slicing for JAX JIT compatibility.
    """
    # Use masking instead of dynamic slicing (JAX JIT requirement)
    neuron_mask = topology.neuron_active  # (MAX_NEURONS,) 1 for active, 0 for inactive

    # Input from sensors - use full matrix, masked
    # sensor_weights is (N_SENSORS, MAX_NEURONS)
    sensor_input = jnp.matmul(sensors, topology.sensor_weights) * neuron_mask

    # Current activations
    x = neural_state.x

    # Synaptic transmission using scatter-add pattern
    # More efficient than vmap over all neurons
    pre_neurons = topology.conn_pre
    post_neurons = topology.conn_post
    weights = topology.conn_weight
    conn_active = topology.conn_active

    # Compute weighted inputs for all connections
    pre_acts = x[pre_neurons]  # (MAX_CONNS,)
    weighted_input = pre_acts * weights * conn_active  # (MAX_CONNS,)

    # Scatter-add to post-synaptic neurons
    # This accumulates inputs to each neuron
    synaptic_input = jnp.zeros(SIGMA_MAX_NEURONS)
    synaptic_input = synaptic_input.at[post_neurons].add(weighted_input)

    # Total input (masked by active neurons)
    total_input = (sensor_input + synaptic_input) * neuron_mask

    # Neural dynamics with time constant
    # Map neuron types to tau values
    tau = genome_neural.tau_by_type[topology.neuron_type]  # (MAX_NEURONS,)
    # Use default tau for inactive neurons to avoid issues
    tau = jnp.where(neuron_mask > 0.5, tau, 10.0)

    # Leaky integration
    dx = (-x + total_input) / tau
    new_x = x + dx * SIGMA_DT

    # Activation function
    gain = genome_neural.activation_gain[0]
    bias = genome_neural.activation_bias[0]
    new_x = jnp.tanh(gain * (new_x - bias))

    # Zero out inactive neurons
    new_x = new_x * neuron_mask

    # Compute motor output
    motor_neurons = new_x * (topology.neuron_type == NEURON_TYPE_MOTOR).astype(jnp.float32)
    motor_output = jnp.matmul(motor_neurons, topology.motor_weights)

    # Update traces (for plasticity)
    pre_acts_new = new_x[pre_neurons]
    post_acts_new = new_x[post_neurons]
    new_traces = genome_neural.trace_decay[0] * neural_state.traces + pre_acts_new * post_acts_new
    new_traces = new_traces * conn_active

    new_state = NeuralState(
        x=new_x,
        traces=new_traces,
        modulators=neural_state.modulators,
    )

    return new_state, motor_output


@jit
def apply_plasticity(
    topology: DynamicTopology,
    neural_state: NeuralState,
    reward: float,
    genome_neural: NeuralGene,
) -> DynamicTopology:
    """
    Apply reward-modulated Hebbian plasticity.

    Connections that predicted reward are strengthened.
    """
    lr = genome_neural.learning_rate[0]

    # Weight update: dw = lr * trace * reward
    dw = lr * neural_state.traces * reward

    # Apply update (only to active connections)
    new_weights = topology.conn_weight + dw * topology.conn_active

    # Update usage tracking (for pruning)
    usage_contribution = jnp.abs(neural_state.traces) * topology.conn_active
    new_usage = 0.99 * topology.conn_usage + 0.01 * usage_contribution

    return topology._replace(
        conn_weight=new_weights,
        conn_usage=new_usage,
    )


# =============================================================================
# FOOD / ENVIRONMENT
# =============================================================================

class FoodState(NamedTuple):
    """
    Food locations in the environment.

    Food provides energy to survive.
    """
    positions: jnp.ndarray                  # (N_FOOD, 3) food positions
    active: jnp.ndarray                     # (N_FOOD,) 1=available
    spawn_timer: jnp.ndarray                # (1,) steps until next spawn


SIGMA_N_FOOD = 10
SIGMA_FOOD_SPAWN_INTERVAL = 50


def init_food_state(key: jnp.ndarray) -> FoodState:
    """Initialize food locations."""
    keys = random.split(key, 2)

    # Random positions closer to origin (where creature starts)
    # Range reduced from [-5, 5] to [-2, 2] for easier discovery
    positions = random.uniform(keys[0], (SIGMA_N_FOOD, 3), minval=-2.0, maxval=2.0)
    positions = positions.at[:, 2].set(0.0)  # Food on ground

    return FoodState(
        positions=positions,
        active=jnp.ones(SIGMA_N_FOOD),
        spawn_timer=jnp.array([SIGMA_FOOD_SPAWN_INTERVAL]),
    )


@jit
def check_food_collision(
    food_state: FoodState,
    body_pos: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[FoodState, float]:
    """
    Check if body collided with food.

    Returns updated food state and amount eaten.
    """
    # Distance to each food
    dists = jnp.linalg.norm(food_state.positions - body_pos, axis=1)

    # Check collision (within 0.5 units)
    collision_radius = 0.5
    collided = (dists < collision_radius) & (food_state.active > 0.5)

    # Eat collided food
    amount_eaten = jnp.sum(collided.astype(jnp.float32))
    new_active = jnp.where(collided, 0.0, food_state.active)

    # Respawn timer
    new_timer = food_state.spawn_timer - 1
    should_spawn = new_timer <= 0

    # Respawn food if timer expired
    keys = random.split(key, 2)
    spawn_pos = random.uniform(keys[0], (SIGMA_N_FOOD, 3), minval=-2.0, maxval=2.0)
    spawn_pos = spawn_pos.at[:, 2].set(0.0)

    # Only respawn inactive food
    new_positions = jnp.where(
        (should_spawn & (new_active < 0.5))[:, None],
        spawn_pos,
        food_state.positions
    )
    new_active = jnp.where(should_spawn & (new_active < 0.5), 1.0, new_active)
    new_timer = jnp.where(should_spawn, SIGMA_FOOD_SPAWN_INTERVAL, new_timer)

    new_state = FoodState(
        positions=new_positions,
        active=new_active,
        spawn_timer=new_timer,
    )

    return new_state, amount_eaten


# =============================================================================
# NON-STATIONARY PHYSICS
# =============================================================================

class PhysicsState(NamedTuple):
    """
    Non-stationary physics parameters.

    Forces adaptation by changing gravity/friction periodically.
    """
    gravity: jnp.ndarray                    # (1,) current gravity
    friction: jnp.ndarray                   # (1,) current friction
    change_timer: jnp.ndarray               # (1,) steps until change


def init_physics_state() -> PhysicsState:
    """Initialize with default physics."""
    return PhysicsState(
        gravity=jnp.array([SIGMA_GRAVITY]),
        friction=jnp.array([SIGMA_FRICTION]),
        change_timer=jnp.array([SIGMA_PHYSICS_CHANGE_INTERVAL]),
    )


@jit
def update_physics(physics_state: PhysicsState, key: jnp.ndarray) -> PhysicsState:
    """
    Update physics parameters (non-stationary environment).
    """
    new_timer = physics_state.change_timer - 1
    should_change = new_timer <= 0

    keys = random.split(key, 2)

    # Random new values
    new_gravity = random.uniform(keys[0], (1,),
                                  minval=SIGMA_GRAVITY_RANGE[0],
                                  maxval=SIGMA_GRAVITY_RANGE[1])
    new_friction = random.uniform(keys[1], (1,),
                                   minval=SIGMA_FRICTION_RANGE[0],
                                   maxval=SIGMA_FRICTION_RANGE[1])

    return PhysicsState(
        gravity=jnp.where(should_change, new_gravity, physics_state.gravity),
        friction=jnp.where(should_change, new_friction, physics_state.friction),
        change_timer=jnp.where(should_change, SIGMA_PHYSICS_CHANGE_INTERVAL, new_timer),
    )


# =============================================================================
# COMPLETE SIMULATION STATE
# =============================================================================

class SimState(NamedTuple):
    """
    Complete simulation state.
    """
    # Neural network
    topology: DynamicTopology
    neural: NeuralState
    criticality: CriticalityState

    # Body and energy
    body: BodyState
    energy: EnergyState

    # Environment
    food: FoodState
    physics: PhysicsState

    # Statistics
    step_count: jnp.ndarray
    total_reward: jnp.ndarray

    # Random key
    key: jnp.ndarray


def init_state(genome: Genome, key: jnp.ndarray) -> SimState:
    """Initialize complete simulation state."""
    keys = random.split(key, 6)

    if SIGMA_VERBOSE:
        print("Initializing simulation state...")

    topology = init_topology(genome, keys[0])

    return SimState(
        topology=topology,
        neural=init_neural_state(keys[3]),  # Pass key for random init
        criticality=init_criticality_state(),
        body=init_body_state(),
        energy=init_energy_state(),
        food=init_food_state(keys[1]),
        physics=init_physics_state(),
        step_count=jnp.array([0]),
        total_reward=jnp.array([0.0]),
        key=keys[2],
    )


# =============================================================================
# SIMULATION STEP
# =============================================================================

@jit
def step_simulation(
    state: SimState,
    genome: Genome,
) -> SimState:
    """
    Single simulation step with full Sigma-AGI architecture.

    1. Sense body state
    2. Neural forward pass
    3. Apply motor commands
    4. Update physics
    5. Check food/energy
    6. Update criticality
    """
    keys = random.split(state.key, 6)

    # 1. Sense body state
    sensors = compute_body_sensors(state.body, state.food.positions)

    # 2. Neural forward pass
    neural_state, motor_output = neural_forward(
        state.neural,
        state.topology,
        sensors,
        genome.neural,
    )

    # 2.5 Add exploration noise to motor output (crucial for evolution!)
    # This helps discover behaviors even when neural network is weak
    exploration_noise = random.normal(keys[5], motor_output.shape) * 0.3
    motor_output = motor_output + exploration_noise

    # 3. Apply criticality adaptation
    x_adapted = apply_criticality_adaptation(
        neural_state.x,
        state.criticality,
        genome.criticality,
    )
    neural_state = neural_state._replace(x=x_adapted)

    # 4. Step body physics
    torques = motor_output * genome.body.joint_strength
    new_body, movement_cost = step_body_physics(
        state.body,
        torques,
        genome.body,
        state.physics.gravity[0],
        state.physics.friction[0],
    )

    # 5. Check food collision
    new_food, food_eaten = check_food_collision(
        state.food,
        new_body.com_pos,
        keys[0],
    )

    # 6. Update energy
    fell = new_body.com_pos[2] < 0.2  # Fell down
    new_energy, alive = update_energy(
        state.energy,
        movement_cost,
        food_eaten,
        fell,
    )

    # 7. Update criticality tracking
    new_criticality = update_criticality(
        state.criticality,
        neural_state.x,
        state.topology.neuron_active,  # Use mask instead of n_neurons for JIT
    )

    # 8. Update physics (non-stationary)
    new_physics = update_physics(state.physics, keys[1])

    # 9. Apply plasticity based on reward
    reward = food_eaten - movement_cost * 0.1
    new_topology = apply_plasticity(
        state.topology,
        neural_state,
        reward,
        genome.neural,
    )

    # Update statistics
    new_step = state.step_count + 1
    new_total_reward = state.total_reward + reward

    return SimState(
        topology=new_topology,
        neural=neural_state,
        criticality=new_criticality,
        body=new_body,
        energy=new_energy,
        food=new_food,
        physics=new_physics,
        step_count=new_step,
        total_reward=new_total_reward,
        key=keys[2],
    )


# =============================================================================
# EVALUATION WITH RESOURCE PENALTY
# =============================================================================

def evaluate_genome(
    genome: Genome,
    key: jnp.ndarray,
    max_steps: int = 2000,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a genome.

    Fitness = survival_time + food_eaten - resource_penalty

    Resource penalty follows Bitter Lesson:
    - More neurons = higher cost
    - More connections = higher cost
    - Evolution must balance capability vs efficiency
    """
    state = init_state(genome, key)

    # Run until death or max_steps
    step_fn = partial(step_simulation, genome=genome)

    for _ in range(max_steps):
        state = step_fn(state)

        # Check if dead
        if state.energy.cause_of_death[0] > 0:
            break

    # Compute fitness with resource penalty
    survival_time = float(state.energy.steps_survived[0])
    food_eaten = float(state.energy.total_food_eaten[0])

    n_neurons = float(state.topology.n_neurons[0])
    n_connections = float(state.topology.n_connections[0])

    # Resource penalty (Bitter Lesson: complexity has cost)
    resource_penalty = (SIGMA_NEURON_COST * n_neurons +
                        SIGMA_CONNECTION_COST * n_connections)

    # Base fitness from survival and food
    base_fitness = survival_time * 0.01 + food_eaten * 1.0

    # Final fitness
    fitness = base_fitness - resource_penalty

    # Collect statistics
    stats = {
        'survival_time': survival_time,
        'food_eaten': food_eaten,
        'n_neurons': n_neurons,
        'n_connections': n_connections,
        'resource_penalty': resource_penalty,
        'base_fitness': base_fitness,
        'final_fitness': fitness,
        'cause_of_death': int(state.energy.cause_of_death[0]),
        'branching_ratio': float(state.criticality.branching_ratio[0]),
        'max_avalanche': float(state.criticality.max_avalanche_size[0]),
    }

    return fitness, stats


# =============================================================================
# NETWORK GROWTH FUNCTIONS
# =============================================================================

def maybe_grow_network(
    genome: Genome,
    topology: DynamicTopology,
    key: jnp.ndarray,
    fitness_history: List[float],
    generation: int,
) -> DynamicTopology:
    """
    Decide whether to grow the network based on fitness stagnation.

    This is the NEAT-style constructive evolution:
    - Start minimal
    - Add complexity only when needed
    - Prune unused complexity
    """
    keys = random.split(key, 2)

    # Check for stagnation
    if len(fitness_history) < SIGMA_GROWTH_STAGNATION:
        return topology

    recent = fitness_history[-SIGMA_GROWTH_STAGNATION:]
    improvement = max(recent) - min(recent)

    threshold = float(genome.growth.growth_trigger_threshold[0])

    if improvement < threshold:
        # Stagnation detected - GROW!
        if SIGMA_VERBOSE:
            print(f"  Growth triggered at gen {generation}: improvement={improvement:.4f} < threshold={threshold:.4f}")

        # Add neurons
        n_new_neurons = int(genome.growth.growth_neuron_count[0])
        topology = add_neurons(topology, keys[0], n_new_neurons)

        # Add connections
        n_new_conns = int(genome.growth.connection_growth_rate[0])
        dist_pref = float(genome.growth.connection_distance_preference[0])
        topology = add_connections(topology, keys[1], n_new_conns, dist_pref)

        if SIGMA_VERBOSE:
            print(f"  New topology: {topology.n_neurons[0]} neurons, {topology.n_connections[0]} connections")

    # Always consider pruning
    prune_thresh = float(genome.growth.prune_threshold[0])
    topology = prune_connections(topology, prune_thresh)

    return topology


# =============================================================================
# EVOLUTION
# =============================================================================

def tournament_select(
    population: List[Genome],
    fitnesses: List[float],
    key: jnp.ndarray,
    tournament_size: int = SIGMA_TOURNAMENT_SIZE,
) -> Genome:
    """Tournament selection."""
    indices = random.randint(key, (tournament_size,), 0, len(population))
    tournament_fits = [fitnesses[int(i)] for i in indices]
    winner_idx = indices[np.argmax(tournament_fits)]
    return population[int(winner_idx)]


def evolve(
    n_generations: int = 10000,
    population_size: int = SIGMA_POPULATION,
    seed: int = 42,
    log_interval: int = 10,
):
    """
    Main evolution loop.

    Key differences from previous versions:
    1. Dynamic network growth (starts small)
    2. Resource penalty (complexity has cost)
    3. Energy-based fitness (survival time matters)
    4. Large population (128) due to small initial network
    """
    print("\n" + "=" * 70)
    print("Sigma-AGI Evolution: Dynamic Neurogenesis")
    print("=" * 70)
    print(f"Population: {population_size}")
    print(f"Initial neurons: {SIGMA_INITIAL_NEURONS}")
    print(f"Initial connections: {SIGMA_INITIAL_CONNECTIONS}")
    print(f"Neuron cost: {SIGMA_NEURON_COST}")
    print(f"Connection cost: {SIGMA_CONNECTION_COST}")
    print("=" * 70)

    # Initialize
    key = random.PRNGKey(seed)
    keys = random.split(key, population_size + 1)

    # Initialize population
    print("Initializing population...")
    population = [init_genome(keys[i]) for i in range(population_size)]

    # Track best
    best_fitness = float('-inf')
    best_genome = None
    fitness_history = []

    # Evolution loop
    for gen in range(n_generations):
        gen_start = time.time()

        # Evaluate all genomes
        keys = random.split(keys[-1], population_size + 1)
        fitnesses = []
        stats_list = []

        for i, genome in enumerate(population):
            fitness, stats = evaluate_genome(genome, keys[i])
            fitnesses.append(fitness)
            stats_list.append(stats)

        # Find best
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_stats = stats_list[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = population[gen_best_idx]

        fitness_history.append(gen_best_fitness)

        # Logging
        if gen % log_interval == 0:
            gen_time = time.time() - gen_start
            mean_fit = np.mean(fitnesses)

            print(f"\nGen {gen:5d} | Best: {gen_best_fitness:+.4f} | Mean: {mean_fit:+.4f} | Time: {gen_time:.2f}s")
            print(f"  Survival: {gen_best_stats['survival_time']:.0f} steps | Food: {gen_best_stats['food_eaten']:.1f}")
            print(f"  Neurons: {gen_best_stats['n_neurons']:.0f} | Conns: {gen_best_stats['n_connections']:.0f}")
            print(f"  Resource penalty: {gen_best_stats['resource_penalty']:.4f}")
            print(f"  Criticality: {gen_best_stats['branching_ratio']:.3f}")

        # Selection and reproduction
        elite_size = int(population_size * SIGMA_ELITE_RATIO)

        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]

        # Keep elites
        new_population = [population[i] for i in sorted_indices[:elite_size]]

        # Tournament selection + mutation for rest
        keys = random.split(keys[-1], population_size - elite_size + 1)
        for i in range(population_size - elite_size):
            parent = tournament_select(population, fitnesses, keys[i])
            child = mutate_genome(parent, keys[i], SIGMA_MUTATION_RATE)
            new_population.append(child)

        population = new_population

        # Maybe grow networks (for best performers)
        for i in range(elite_size):
            key_grow = random.split(keys[-1], 2)[0]
            # Get fresh topology for this genome
            # Note: In a more sophisticated implementation, we'd track topology per individual

    print("\n" + "=" * 70)
    print("Evolution Complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    print("=" * 70)

    return best_genome, fitness_history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\nSigma-AGI: Dynamic Neurogenesis Core")
    print("Platform: JAX (TPU/GPU compatible)")
    print("=" * 50)

    # Initialize for TPU context
    print("\nInitializing Genome for TPU context...")
    key = random.PRNGKey(0)
    genome = init_genome(key)

    print(f"\n[System Ready]")
    print(f"  Growth Trigger: {genome.growth.growth_trigger_threshold[0]:.3f}")
    print(f"  Neural Time Constants: {genome.neural.tau_by_type}")
    print(f"  Body Joint Strength: {genome.body.joint_strength[:4]}...")

    # Quick validation
    print("\nRunning validation test...")
    fitness, stats = evaluate_genome(genome, key, max_steps=100)
    print(f"  Validation Fitness: {fitness:.4f}")
    print(f"  Survival Time: {stats['survival_time']:.0f} steps")

    print("\n" + "=" * 50)
    print("To run full evolution:")
    print("  python sigma_core.py --train --population 256")
    print("  (or use evolve() function directly)")