#!/usr/bin/env python

""" This is the starter code for the robot localization project """

import rospy

from std_msgs.msg import Header, String
from sensor_msgs.msg import LaserScan, PointCloud
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion, Point32
from nav_msgs.srv import GetMap
from copy import deepcopy
from nav_msgs.msg import OccupancyGrid

import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
from random import gauss

import math
import time

import numpy as np
import scipy
from numpy.random import random_sample
from sklearn.neighbors import NearestNeighbors
from occupancy_field import OccupancyField

from helper_functions import (convert_pose_inverse_transform,
                              convert_translation_rotation_to_pose,
                              convert_pose_to_xy_and_theta,
                              angle_diff)


class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """ Construct a new Particle
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized """
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        orientation_tuple = tf.transformations.quaternion_from_euler(0, 0, self.theta)
        return Pose(position=Point(x=self.x, y=self.y, z=0),
                    orientation=Quaternion(x=orientation_tuple[0], y=orientation_tuple[1], z=orientation_tuple[2],
                                           w=orientation_tuple[3]))

        # TODO: define additional helper functions if needed


class ParticleFilter(object):
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            initialized: a Boolean flag to communicate to other class methods that initializaiton is complete
            base_frame: the name of the robot base coordinate frame (should be "base_link" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            laser_max_distance: the maximum distance to an obstacle we should use in a likelihood calculation
            pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            laser_subscriber: listens for new scan data on topic self.scan_topic
            tf_listener: listener for coordinate transforms
            tf_broadcaster: broadcaster for coordinate transforms
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            map: the map we will be localizing ourselves in.  The map should be of type nav_msgs/OccupancyGrid
    """

    def __init__(self):
        self.initialized = False  # make sure we don't perform updates before everything is setup
        rospy.init_node('pf')  # tell roscore that we are creating a new node named "pf"

        self.base_frame = "base_link"  # the frame of the robot base
        self.map_frame = "map"  # the name of the map coordinate frame
        self.odom_frame = "odom"  # the name of the odometry coordinate frame
        self.scan_topic = "scan"  # the topic where we will get laser scans from

        self.n_particles = 300  # the number of particles to use

        self.d_thresh = 0.2  # the amount of linear movement before performing an update
        self.a_thresh = math.pi / 6  # the amount of angular movement before performing an update

        self.laser_max_distance = 2.0  # maximum penalty to assess in the likelihood field model

        # TODO: define additional constants if needed

        # Setup pubs and subs

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.pose_listener = rospy.Subscriber("initialpose", PoseWithCovarianceStamped, self.update_initial_pose)
        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = rospy.Publisher("particlecloud", PoseArray, queue_size=10)

        # laser_subscriber listens for data from the lidar
        self.laser_subscriber = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_received)

        # enable listening for and broadcasting coordinate transforms
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

        self.particle_cloud = []

        self.current_odom_xy_theta = []

        # Make a ros service call to the /static_map service to get a nav_msgs/OccupancyGrid map.
        # Then use OccupancyField to make the map object

        robotMap = rospy.ServiceProxy('/static_map', GetMap)().map
        self.occupancy_field = OccupancyField(robotMap)
        print "OccupancyField initialized", self.occupancy_field

        self.initialized = True

    def update_robot_pose(self):
        """ Update the estimate of the robot's pose given the updated particles.
            There are two logical methods for this:
                (1): compute the mean pose
                (2): compute the most likely pose (i.e. the mode of the distribution)

            Our strategy is #2 to enable better tracking of unlikely particles in the future
        """
        # first make sure that the particle weights are normalized
        self.normalize_particles()

        chosen_one = max(self.particle_cloud, key=lambda p: p.w)
        # TODO: assign the lastest pose into self.robot_pose as a geometry_msgs.Pose object
        # just to get started we will fix the robot's pose to always be at the origin
        self.robot_pose = chosen_one.as_pose()

    def update_particles_with_odom(self, msg):
        """ Update the particles using the newly given odometry pose.
            The function computes the value delta which is a tuple (x,y,theta)
            that indicates the change in position and angle between the odometry
            when the particles were last updated and the current odometry.

            msg: this is not really needed to implement this, but is here just in case.
        """
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     angle_diff(new_odom_xy_theta[2], self.current_odom_xy_theta[2]))

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta
            return


        for i, particle in enumerate(self.particle_cloud):
            # TODO: Change odometry uncertainty to be ROS param

            # Calculate the angle difference between the old odometry position
            # and the old particle position. Then create a rotation matrix between
            # the two angles
            rotationmatrix = self.make_rotation_matrix(particle.theta - old_odom_xy_theta[2])

            # rotate the motion vector, add the result to the particle
            rotated_delta = np.dot(rotationmatrix, delta[:2])

            linear_randomness = np.random.normal(1, 0.2)
            angular_randomness = np.random.uniform(1, 0.3)

            particle.x += rotated_delta[0] * linear_randomness
            particle.y += rotated_delta[1] * linear_randomness

            particle.theta += delta[2] * angular_randomness

            # Make sure the particle's angle doesn't wrap
            particle.theta = angle_diff(particle.theta, 0)

    def make_rotation_matrix(self, theta):
        """ make_rotation_matrix returns a rotation matrix given angle theta

        Args:
            theta (number): the angle of rotation in radians CCW

        Returns:
            ndarray: a two by two rotation matrix

        """
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)

        return np.array([[cosTheta, -sinTheta],
                         [sinTheta, cosTheta]])



    def map_calc_range(self, x, y, theta):
        """ Difficulty Level 3: implement a ray tracing likelihood model... Let me know if you are interested """
        # TODO: nothing unless you want to try this alternate likelihood model
        pass

    def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample.
        """
        # make sure the distribution is normalized
        self.normalize_particles()

        choices = self.particle_cloud
        probabilities = [p.w for p in choices]
        # TODO: Dynamically decide how many particles we need
        n = self.n_particles

        new_particles = self.draw_random_sample(choices, probabilities, n)

        # Set all of the weights back to the same value. Concentration of particles reflects weight.
        for p in new_particles:
            p.w = 1.0
        self.normalize_particles()

        self.particle_cloud = new_particles

    @staticmethod
    def laser_uncertainty_model(distErr):
        """
        Computes the probability of the laser returning a point distance distErr from the wall.
        Note that this uses an exponential distribution instead of anything reasonable for computational speed.

        Args:
            distErr (float): The distance between the point returned and the nearest
                            wall on the map (in meters)

        Returns:
            probability (float): A probability, in the range 0...1
        """

        # TODO: make these into rosparams
        k = 0.1  # meters of half-life of distance probability for main distribution
        probMiss = 0.05  # Base probability that the laser scan is totally confused

        distErr = abs(distErr)

        return (1/(1+probMiss)) * (probMiss + 1/(distErr / k + 1))


    def update_particles_with_laser(self, msg):
        """ Updates the particle weights in response to the scan contained in the msg
        Args:
            msg (LaserScan): incoming message
        """

        # Transform to cartesian coordinates
        scan_points = PointCloud()
        scan_points.header= msg.header

        for i, range in enumerate(msg.ranges):
            if range == 0:
                continue
            # Calculate point in laser coordinate frame
            angle = msg.angle_min + i*msg.angle_increment
            x = range * np.cos(angle)
            y = range * np.sin(angle)
            scan_points.points.append(Point32(x=x, y=y))

        # Transform into base_link coordinates
        scan_points = self.tf_listener.transformPointCloud('base_link', scan_points)

        # For each particle...
        for particle in self.particle_cloud:

            # Create a 3x3 matrix that transforms points from the origin to the particle
            rotmatrix = np.matrix([[np.cos(particle.theta), -np.sin(particle.theta), 0],
                                   [np.sin(particle.theta), np.cos(particle.theta), 0],
                                   [0, 0, 1]])
            transmatrix = np.matrix([[1, 0, particle.x],
                                     [0, 1, particle.y],
                                     [0, 0, 1]])
            mat33 = np.dot(transmatrix, rotmatrix)

            # Iterate through the points in the laser scan

            probabilities = []
            for point in scan_points.points:
                # Move the point onto the particle
                xy = np.dot(mat33, np.array([point.x, point.y, 1]))

                # Figure out the probability of that point
                distToWall = self.occupancy_field.get_closest_obstacle_distance(xy.item(0), xy.item(1))
                if np.isnan(distToWall):
                    continue

                probabilities.append(self.laser_uncertainty_model(distToWall))

            # Combine those into probability of this scan given hypothesized location
            # This is the bullshit thing Paul showed
            # TODO: exponent should be a rosparam
            totalProb = np.sum([p ** 3 for p in probabilities]) / len(probabilities)

            # Update the particle's probability with new info

            particle.w *= totalProb

        # Normalize particles
        self.normalize_particles()


    @staticmethod
    def weighted_values(values, probabilities, size):
        """ Return a random sample of size elements from the set values with the specified probabilities
            values: the values to sample from (numpy.ndarray)
            probabilities: the probability of selecting each element in values (numpy.ndarray)
            size: the number of samples
        """
        bins = np.add.accumulate(probabilities)
        return values[np.digitize(random_sample(size), bins)]

    @staticmethod
    def draw_random_sample(choices, probabilities, n):
        """ Return a random sample of n elements from the set choices with the specified probabilities
            Args:
                choices: the values to sample from represented as a list
                probabilities: the probability of selecting each element in choices represented as a list
                n: the number of samples

            Returns:
                samples (List): A list of n elements, deep-copied from choices
        """
        values = np.array(range(len(choices)))
        probs = np.array(probabilities)
        bins = np.add.accumulate(probs)
        inds = values[np.digitize(random_sample(n), bins)]
        samples = []
        for i in inds:
            samples.append(deepcopy(choices[int(i)]))
        return samples

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        xy_theta = convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(xy_theta)
        self.fix_map_to_odom_transform(msg)

    def initialize_particle_cloud(self, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is ommitted, the odometry will be used """
        if xy_theta is None:
            xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)
        self.particle_cloud = []

        linear_variance = 0.5  # meters
        angular_variance = 4

        xs = np.random.normal(xy_theta[0], linear_variance, size=self.n_particles)
        ys = np.random.normal(xy_theta[1], linear_variance, size=self.n_particles)
        thetas = np.random.vonmises(xy_theta[2], angular_variance, size=self.n_particles)

        self.particle_cloud = [Particle(x=xs[i], y=ys[i], theta=thetas[i]) for i in xrange(self.n_particles)]

        self.normalize_particles()
        self.update_robot_pose()

    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """

        total = sum([p.w for p in self.particle_cloud])

        if total != 0:
            for p in self.particle_cloud:
                p.w /= total

        # Plan: divide each by the sum of all
        # TODO: implement this

    def publish_particles(self, msg):
        particles_conv = []
        for p in self.particle_cloud:
            particles_conv.append(p.as_pose())
        # actually send the message so that we can view it in rviz
        self.particle_pub.publish(PoseArray(header=Header(stamp=rospy.Time.now(),
                                                          frame_id=self.map_frame),
                                            poses=particles_conv))

    def scan_received(self, msg):
        """ This is the default logic for what to do when processing scan data.
            Feel free to modify this, however, I hope it will provide a good
            guide.  The input msg is an object of type sensor_msgs/LaserScan """
        if not (self.initialized):
            # wait for initialization to complete
            return

        if not (self.tf_listener.canTransform(self.base_frame, msg.header.frame_id, msg.header.stamp)):
            # need to know how to transform the laser to the base frame
            # this will be given by either Gazebo or neato_node
            return

        if not (self.tf_listener.canTransform(self.base_frame, self.odom_frame, msg.header.stamp)):
            # need to know how to transform between base and odometric frames
            # this will eventually be published by either Gazebo or neato_node
            return

        # calculate pose of laser relative ot the robot base
        p = PoseStamped(header=Header(stamp=rospy.Time(0),
                                      frame_id=msg.header.frame_id))
        self.laser_pose = self.tf_listener.transformPose(self.base_frame, p)

        # find out where the robot thinks it is based on its odometry
        p = PoseStamped(header=Header(stamp=msg.header.stamp,
                                      frame_id=self.base_frame),
                        pose=Pose())
        self.odom_pose = self.tf_listener.transformPose(self.odom_frame, p)
        # store the the odometry pose in a more convenient format (x,y,theta)
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)

        if not (self.particle_cloud):
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud()
            # cache the last odometric pose so we can only update our particle filter if we move more than self.d_thresh or self.a_thresh
            self.current_odom_xy_theta = new_odom_xy_theta
            # update our map to odom transform now that the particles are initialized
            self.fix_map_to_odom_transform(msg)
        elif (math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or
                      math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or
                      math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh):
            # we have moved far enough to do an update!
            self.update_particles_with_odom(msg)  # update based on odometry
            self.update_particles_with_laser(msg)  # update based on laser scan
            self.update_robot_pose()  # update robot's pose
            self.resample_particles()  # resample particles to focus on areas of high density
            self.fix_map_to_odom_transform(msg)  # update map to odom transform now that we have new particles
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg)

    def fix_map_to_odom_transform(self, msg):
        """ This method constantly updates the offset of the map and
            odometry coordinate systems based on the latest results from
            the localizer
            TODO: if you want to learn a lot about tf, reimplement this... I can provide
                  you with some hints as to what is going on here. """
        (translation, rotation) = convert_pose_inverse_transform(self.robot_pose)
        p = PoseStamped(pose=convert_translation_rotation_to_pose(translation, rotation),
                        header=Header(stamp=msg.header.stamp, frame_id=self.base_frame))
        self.tf_listener.waitForTransform(self.base_frame, self.odom_frame, msg.header.stamp, rospy.Duration(1.0))
        self.odom_to_map = self.tf_listener.transformPose(self.odom_frame, p)
        (self.translation, self.rotation) = convert_pose_inverse_transform(self.odom_to_map.pose)

    def broadcast_last_transform(self):
        """ Make sure that we are always broadcasting the last map
            to odom transformation.  This is necessary so things like
            move_base can work properly. """
        if not (hasattr(self, 'translation') and hasattr(self, 'rotation')):
            return
        self.tf_broadcaster.sendTransform(self.translation,
                                          self.rotation,
                                          rospy.get_rostime(),
                                          self.odom_frame,
                                          self.map_frame)


if __name__ == '__main__':
    n = ParticleFilter()
    r = rospy.Rate(5)

    while not (rospy.is_shutdown()):
        # in the main loop all we do is continuously broadcast the latest map to odom transform
        n.broadcast_last_transform()
        r.sleep()
