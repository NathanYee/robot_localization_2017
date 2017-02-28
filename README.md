# Robot Localization

## Computational Robotics Spring 2017

## Eric Miller, Nathan Yee

<!---
Gif of AC109_2 turning corner - pretty laser scans
-->
![](https://github.com/NathanYee/robot_localization_2017/blob/master/my_localizer/media/good2.gif)

# Robot Localization and Particle Filters

Robot localization is the process of a robot autonomously determining its position in a pre-defined map. Our solution to the localization task is through the means of a [particle filter](https://en.wikipedia.org/wiki/Particle_filter), which involves a combination of [Monte Carlo methodology](https://en.wikipedia.org/wiki/Monte_Carlo_method), [Bayesian Inference](https://en.wikipedia.org/wiki/Bayesian_inference), and sensor processing.

# Basic Architecture

Our particle filter begins by initializing particles normally distributed around the robot's position. This is important as it allows us to give the robot an accurate initial position for debugging purposes.

On receiving odometry updates, the particles move in tandem with the update multiplied by a small amount of randomness.

On receiving a laser scan, the particle filter determines how likely that each particle represents the true position of the robot.

After the laser scan update, the particle filter determines a best guess for the position of the robot.

Finally, the particle filter resamples the particles to restart the cycle.

# Lost factor

## Concept

One common limitation of particle filters is an inability to solve the "lost robot" or “kidnapped robot” problems because all of the particles can become tightly clustered around one incorrect hypothesis. To address this, we implemented a lost-robot correction system which runs during resampling and uses a portion of the particles to attempt to discover other plausible hypothesis of robot state. The portion of particles to be used in this way is referred to as `p_lost` and reflects the filter's estimate of the probability that the robot is in a position not well covered by the particles currently alive. Currently, `p_lost = 0.3`, but the correctness of the filter seems to be very insensitive to that value, so little tuning has been needed. 

## Spawning frontier particles

	new_outliers = outliers[:self.outliers_to_keep] + \
	[Particle().generate_uniformly_on_map(self.occupancy_field.map) for _ in xrange(num_to_make)]

At each resampling of the filter, 70% of the particles are spawned in the normal way, with a weighted random sampling over all of the existing particles. The other 30% are generated as "frontier particles", and are placed in two ways.

* 50% of the outliers detected in the previous step survive as "plausible hypotheses" to prevent premature termination of a reasonable guess

* The remainder of the frontier particles are randomly generated somewhere in the map, with no regard for other previously-existing particles

This system is effectively able to track robot pose if given a reasonable initial estimate, but can recover the pose of an uninitialized or incorrectly-initialized robot after less than a minute (depending on the size and structure of the mapped area).

## Outlier detection

We use unsupervised outlier detection to determine which particles we want to consider outliers which ultimately become "frontier particles". Our choice of algorithm is Scikit-Learn's One-Class SVM because it is highly optimized and allows for precise control of the number of particles to be considered outliers.

# Turn multiplier estimation

One issue we noticed during testing was a perceived systematic bias in odometry turn measurements. This corrective factor seemed to range between 1.3 and 1.8, with different values applying on different data files. While we have not conclusively determined the source of this error, we have altered our filter to effectively estimate and correct for its effects. 

To do this, we altered each particle to track an additional piece of robot state on top of the pose of the robot: the magnitude of the turn compensation. This is estimated as a joint distribution with the pose, allowing the filter to use multiple measurements to contribute to its estimate.

When a new particle is spawned, it takes the turn multiplier estimate of its source, mutated by a small amount of randomness. If a frontier particle is being spawned without a source, it instead takes the filter's "best estimate" of the true turn multiplier, currently calculated as the mean of all the particles in a cluster.

# Next steps

## Dynamic estimation of lost factor

The convergence time of the filter could be decreased by dynamically adjusting `p_lost`, starting near 1 if the filter has not been initialized, and decreasing or increasing as appropriate when sensor data is obtained. A few mechanisms are possible, ranging from heuristic observation of whether the most likely pose was generated randomly to a proper bayesian estimate based on the relative probabilities of in-cluster and frontier particles.

## Spread particles when spawned

Currently, the main source of randomness in particle motion comes from an estimate of odometry error. While this does a good job of tracking the robot, it can do a bad job of quickly discovering new locations near to existing particles that are better hypotheses. To fix this, we could intentionally spawn particles with some randomness, rather than using a simple `copy.deepcopy()`.

## Better parameter tuning

The particle filter has yet to be tuned. As a result, it does not behave nearly as well as it could.

## Speed up nearest obstacle data structure

The particle filter currently performs in O(n*m) when looking up the m positions in the map for n particles. This operation can probably be vectorized into linear time.

Or rather than thinking of the map in terms of discrete points, we could think of the map as a series of lines. Then we would only need to determine the distance from each particle to the 6 or so lines.

## Localize while stationary

The particle filter waits until it moves a certain amount before resampling and updating particle positions. It would be advantageous to check new particles during low compute cycles (while the robot is not moving).

## More intelligent frontier particle spawning

The number of frontier particles is currently capped. It would be advantageous to increase the number of frontier particles when the robot thinks it is lost. We would also decrease the number of frontier particles when the robot thinks it has found its correct location.

# Videos

[https://www.youtube.com/watch?v=qAO-hBaivms](https://www.youtube.com/watch?v=qAO-hBaivms)

https://www.youtube.com/watch?v=EgyoL-xq3zc

# For internal development

# MVP Tasks

* Make .gitignore

* Call the map service to get a map

* normalize_particles(self): 

* initialize_particle_cloud(): create particles

* update_particles_with_odom(self, msg): modify particles using delta

* Add randomness to motion

* update_particles_with_laser(self, msg): Implement this

    * Probability of single laser scan point

    * Combining multiple scan point probabilities together

* update_robot_pose(self): assign the lastest pose into self.robot_pose as a geometry_msgs.Pose object

* resample_particles(self): fill out the rest of the implementation

# Extension Tasks

* Use clustering to encourage particles far from center of pose

    * "Lost factor"

* Implement ray tracing map_calc_range()

* Clustering algorithm for finding robot pose

* Covariance matrix for odom uncertainty

* Dynamic reconfigure

* Computational speed analysis

* Allow particles to spread straight sideways
