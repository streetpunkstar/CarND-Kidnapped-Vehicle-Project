/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 100;
  is_initialized = false;

  if (!is_initialized) {
    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (unsigned int i = 0; i < num_particles; ++i) {

      Particle p;

      p.id = i;
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
      p.weight = 1.0;

      particles.push_back(p);
      weights.push_back(1);

    }

    is_initialized = true;

    return;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;

  for (unsigned int j = 0; j < num_particles; ++j) {

    if (fabs(yaw_rate) < 0.00001) {
      particles[j].x += velocity * delta_t * cos(particles[j].theta);
      particles[j].y += velocity * delta_t * sin(particles[j].theta);
    } else {
      particles[j].x += velocity/yaw_rate * (sin(particles[j].theta + yaw_rate * delta_t) - sin(particles[j].theta));
      particles[j].y += velocity/yaw_rate * (cos(particles[j].theta) - cos(particles[j].theta + yaw_rate * delta_t));
      particles[j].theta += yaw_rate * delta_t;
    }

    normal_distribution<double> dist_x(particles[j].x, std_pos[0]);
    normal_distribution<double> dist_y(particles[j].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[j].theta, std_pos[2]);

    particles[j].x = dist_x(gen);
    particles[j].y = dist_y(gen);
    particles[j].theta = dist_theta(gen);

  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  double min_dist = 1.7*pow(10, 308);
  double new_dist;
  int min_id;

  for (unsigned int i = 0; i < observations.size(); ++i) {
    for (unsigned int j = 0; j < predicted.size(); ++j) {

      new_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (new_dist < min_dist) {
        min_dist = new_dist;
        min_id = predicted[j].id; }

    } // end predicted loop
    observations[i].id = min_id;
    // cout << "id closest: " << observations[i].id << endl;

  } // end observations loop
}

double ParticleFilter::MultiGaussProb(double sigma[], double x_obs, double y_obs, double x_mu, double y_mu) {

  double sig_x = sigma[0];
  double sig_y = sigma[1];
  double gauss_norm = 1/(2 * M_PI * sig_x * sig_y);
  double exponent = pow((x_obs - x_mu), 2)/(2 * pow(sig_x, 2)) + pow((y_obs - y_mu), 2)/(2 * pow(sig_y, 2));

  return gauss_norm * exp(-exponent);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

  for (unsigned int r = 0; r < num_particles; ++r) { // PARTICLE LEVEL

    // create predicted vector
    vector<LandmarkObs> predicted;

    for (unsigned int s = 0; s < map_landmarks.landmark_list.size(); ++s) { // LANDMARK LEVEL
      double x_0 = map_landmarks.landmark_list[s].x_f;
      double y_0 = map_landmarks.landmark_list[s].y_f;
      double dist_to_landmark = dist(x_0, y_0, particles[r].x, particles[r].y);
      if (dist_to_landmark <= sensor_range) {
        LandmarkObs q;

        q.x = x_0;
        q.y = y_0;
        q.id = map_landmarks.landmark_list[s].id_i;

        predicted.push_back(q);
      }

    } // END LANDMARK LEVEL

    // convert observations into map coordinates
    vector<LandmarkObs> observations_converted;

    for (unsigned int k = 0; k < observations.size(); ++k) {
      LandmarkObs o;

      o.x = particles[r].x + observations[k].x * cos(particles[r].theta) - observations[k].y * sin(particles[r].theta);
      o.y = particles[r].y + observations[k].x * sin(particles[r].theta) + observations[k].y * cos(particles[r].theta);
      o.id = observations[k].id;

      observations_converted.push_back(o);
    }

    // dataAssociation
    dataAssociation(predicted, observations_converted);
    particles[r].weight = 1.0;

    // update weights
    for (unsigned int t = 0; t < observations_converted.size(); ++t) {

      double measurement_x = observations_converted[t].x;
      double measurement_y = observations_converted[t].y;
      double nearest_x, nearest_y;

      for (unsigned int u = 0; u < predicted.size(); ++u) {
        if (predicted[u].id == observations_converted[t].id) {
          nearest_x = predicted[u].x;
          nearest_y = predicted[u].y;
        }
      }

      double multigauss = MultiGaussProb(std_landmark, measurement_x, measurement_y, nearest_x, nearest_y);

      if (multigauss != 0) {
        particles[r].weight *= multigauss;
      }
    }

  } // END PARTICLE LEVEL

}

void ParticleFilter::resample()
{
  default_random_engine gen;
  vector<double> pdist;
  double total = 0;
  for (int i = 0; i < num_particles; i++)
  {
    total += particles[i].weight;
    pdist.push_back(particles[i].weight);
  }

  // Create New distribution
  discrete_distribution<int> ddist(pdist.begin(), pdist.end());

  // Resample according to the weights
  vector<Particle> resampled_particles(num_particles);
  for (int i = 0; i < num_particles; i++)
  {
    resampled_particles[i] = particles[ddist(gen)];
  }
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
