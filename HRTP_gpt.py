import math
import random
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation

    def move(self, step_size):
        dx = step_size * math.cos(self.orientation)
        dy = step_size * math.sin(self.orientation)
        self.x += dx
        self.y += dy

    def tumble(self):
        self.orientation = random.uniform(0, 2 * math.pi)

def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def attractive_interaction(distance):
    if distance >= 0.5 and distance <= 1:
        return J
    return 0

def energy_difference(old_particles, new_particles):
    energy_diff = 0
    for old_particle, new_particle in zip(old_particles, new_particles):
        dist = distance(old_particle, new_particle)
        interaction = attractive_interaction(dist)
        energy_diff += interaction
        
    return energy_diff

def acceptance_probability(energy_diff):
    return math.exp(-J * energy_diff)

def visualize_particles(particles):
    plt.figure(figsize=(8, 8))
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    for particle in particles:
        plt.arrow(
            particle.x, particle.y,
            0.5 * math.cos(particle.orientation), 0.5 * math.sin(particle.orientation),
            head_width=0.3, head_length=0.3, fc='r', ec='r'
        )
    plt.show()

def update_particles(particles):
    old_particles = particles.copy()
    for particle in particles:
        if random.random() < w:
            particle.tumble()
        else:
            step_size = 0.25
            proposed_particle = Particle(particle.x, particle.y, particle.orientation)
            proposed_particle.move(step_size)

            energy_diff = energy_difference([particle], [proposed_particle])
            accept_prob = acceptance_probability(energy_diff)

            if random.random() < accept_prob:
                particle.x = proposed_particle.x
                particle.y = proposed_particle.y
                particle.orientation = proposed_particle.orientation

    # Check for particle-particle interaction and skip movement if too close
    new_particles = []
    for i in range(len(particles)):
        skip_movement = False
        for j in range(i+1, len(particles)):
            dist = distance(particles[i], particles[j])
            if dist < 0.5:
                skip_movement = True
                break
        if not skip_movement:
            new_particles.append(particles[i])

    return new_particles

def simulate_particles(N, w, J):
    particles = []
    for _ in range(N):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        orientation = random.uniform(0, 2 * math.pi)
        particles.append(Particle(x, y, orientation))

    for _ in range(1000):
        old_particles = particles.copy()

        particles = update_particles(particles)

        # Visualize particles after each Monte Carlo step
        visualize_particles(particles)

        if len(particles) < 2:
            break

    return particles

# Parameters
N = 10  # Number of particles
w = 0.1  # Tumbling rate
J = 1.0  # Attractive interaction strength

# Simulate particles
particles = simulate_particles(N, w, J)
