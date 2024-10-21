#pragma once

// ROS
#include <rclcpp/rclcpp.hpp>
#include <random_numbers/random_numbers.h>

// ROS msgs
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/srv/get_position_fk.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <moveit_msgs/msg/kinematic_solver_info.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>

// KDL
#include <kdl/config.h>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainiksolver.hpp>

// MoveIt
#include <moveit/kinematics_base/kinematics_base.h>
#include <kdl_kinematics_2/joint_mimic.hpp>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#include <cfloat>
#include <string>
#include <vector>

namespace KDL
{
class ChainIkSolverVelMimicSVD;
}

namespace kdl_kinematics_2
{
/**
 * @brief Specific implementation of kinematics using KDL.
 * This version supports any kinematic chain, also including mimic joints.
 */
class KDLKinematics2 : public kinematics::KinematicsBase
{
public:
  /**
   *  @brief Default constructor
   */
  KDLKinematics2();

  bool
  getPositionIK(const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state,
                std::vector<double>& solution, moveit_msgs::msg::MoveItErrorCodes& error_code,
                const kinematics::KinematicsQueryOptions& options = kinematics::KinematicsQueryOptions()) const override;

  bool searchPositionIK(
      const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state, double timeout,
      std::vector<double>& solution, moveit_msgs::msg::MoveItErrorCodes& error_code,
      const kinematics::KinematicsQueryOptions& options = kinematics::KinematicsQueryOptions()) const override;

  bool searchPositionIK(
      const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state, double timeout,
      const std::vector<double>& consistency_limits, std::vector<double>& solution,
      moveit_msgs::msg::MoveItErrorCodes& error_code,
      const kinematics::KinematicsQueryOptions& options = kinematics::KinematicsQueryOptions()) const override;

  bool searchPositionIK(
      const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state, double timeout,
      std::vector<double>& solution, const IKCallbackFn& solution_callback,
      moveit_msgs::msg::MoveItErrorCodes& error_code,
      const kinematics::KinematicsQueryOptions& options = kinematics::KinematicsQueryOptions()) const override;

  bool searchPositionIK(
      const geometry_msgs::msg::Pose& ik_pose, const std::vector<double>& ik_seed_state, double timeout,
      const std::vector<double>& consistency_limits, std::vector<double>& solution,
      const IKCallbackFn& solution_callback, moveit_msgs::msg::MoveItErrorCodes& error_code,
      const kinematics::KinematicsQueryOptions& options = kinematics::KinematicsQueryOptions()) const override;

  bool getPositionFK(const std::vector<std::string>& link_names, const std::vector<double>& joint_angles,
                     std::vector<geometry_msgs::msg::Pose>& poses) const override;

  bool initialize(const rclcpp::Node::SharedPtr& node, const moveit::core::RobotModel& robot_model,
                  const std::string& group_name, const std::string& base_frame,
                  const std::vector<std::string>& tip_frames, double search_discretization) override;

  /**
   * @brief  Return all the joint names in the order they are used internally
   */
  const std::vector<std::string>& getJointNames() const override;

  /**
   * @brief  Return all the link names in the order they are represented internally
   */
  const std::vector<std::string>& getLinkNames() const override;

protected:
  typedef Eigen::Matrix<double, 6, 1> Twist;

  /// Solve position IK given initial joint values
  // NOLINTNEXTLINE(readability-identifier-naming)
  int CartToJnt(KDL::ChainIkSolverVelMimicSVD& ik_solver, const KDL::JntArray& q_init, const KDL::Frame& p_in,
                KDL::JntArray& q_out, const unsigned int max_iter, const Eigen::VectorXd& joint_weights,
                const Twist& cartesian_weights) const;

private:

template<typename T>
bool lookupParam(const rclcpp::Node::SharedPtr& node, const std::string& param_name, T& param_value, const T& default_value)
{
    node->declare_parameter<T>(param_name, default_value);
    bool success = node->get_parameter(param_name, param_value);
    return success;
}
  
  void getJointWeights();
  bool timedOut(const rclcpp::Time& start_time, double duration) const;

  /** @brief Check whether the solution lies within the consistency limits of the seed state
   *  @param seed_state Seed state
   *  @param consistency_limits
   *  @param solution solution configuration
   *  @return true if check succeeds
   */
  bool checkConsistency(const Eigen::VectorXd& seed_state, const std::vector<double>& consistency_limits,
                        const Eigen::VectorXd& solution) const;

  void getRandomConfiguration(Eigen::VectorXd& jnt_array) const;

  /** @brief Get a random configuration within consistency limits close to the seed state
   *  @param seed_state Seed state
   *  @param consistency_limits
   *  @param jnt_array Returned random configuration
   */
  void getRandomConfiguration(const Eigen::VectorXd& seed_state, const std::vector<double>& consistency_limits,
                              Eigen::VectorXd& jnt_array) const;

  /// clip q_delta such that joint limits will not be violated
  void clipToJointLimits(const KDL::JntArray& q, KDL::JntArray& q_delta, Eigen::ArrayXd& weighting) const;

  static rclcpp::Clock steady_clock_;

  bool initialized_;  ///< Internal variable that indicates whether solver is configured and ready

  unsigned int dimension_;                             ///< Dimension of the group
  moveit_msgs::msg::KinematicSolverInfo solver_info_;  ///< Stores information for the inverse kinematics solver

  const moveit::core::JointModelGroup* joint_model_group_;
  moveit::core::RobotStatePtr state_;
  KDL::Chain kdl_chain_;
  std::unique_ptr<KDL::ChainFkSolverPos> fk_solver_;
  std::vector<JointMimic> mimic_joints_;
  std::vector<double> joint_weights_;
  Eigen::VectorXd joint_min_, joint_max_;  ///< joint limits

  int max_solver_iterations_;
  double epsilon_;
  /** weight of orientation error vs position error
   *
   * < 1.0: orientation has less importance than position
   * > 1.0: orientation has more importance than position
   * = 0.0: perform position-only IK */
  double orientation_vs_position_weight_;
};
}  // namespace kdl_kinematics_2
