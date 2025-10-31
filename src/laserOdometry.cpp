#include <cmath>
#include <mutex>
#include <queue>
#include <map>
#include <vector>
#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"
#include "lo_sam/msg/cloud_info.hpp"


#define DISTORTION 0

using std::placeholders::_1;

class LaserOdometryNode : public rclcpp::Node
{
public:
    LaserOdometryNode()
    : Node("laserOdometry"),
      skipFrameNum(5),
      systemInited(false),
      corner_correspondence(0),
      plane_correspondence(0),
      SCAN_PERIOD(0.1),
      DISTANCE_SQ_THRESHOLD(25),
      NEARBY_SCAN(2.5),
      frameCount(0)
    {
        this->declare_parameter<int>("mapping_skip_frame", 2);
        this->get_parameter("mapping_skip_frame", skipFrameNum);

        subCornerPointsSharp = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "feature/cloud_sharp", 100, std::bind(&LaserOdometryNode::laserCloudSharpHandler, this, _1));
        subCornerPointsLessSharp = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "feature/cloud_less_sharp", 100, std::bind(&LaserOdometryNode::laserCloudLessSharpHandler, this, _1));
        subSurfPointsFlat = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "feature/cloud_flat", 100, std::bind(&LaserOdometryNode::laserCloudFlatHandler, this, _1));
        subSurfPointsLessFlat = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "feature/cloud_less_flat", 100, std::bind(&LaserOdometryNode::laserCloudLessFlatHandler, this, _1));
        subLaserCloudFullRes = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "feature/cloud_full", 100, std::bind(&LaserOdometryNode::laserCloudFullResHandler, this, _1));

        pubLaserCloudCornerLast = this->create_publisher<sensor_msgs::msg::PointCloud2>("odom_init/cloud_edge", 100);
        pubLaserCloudSurfLast = this->create_publisher<sensor_msgs::msg::PointCloud2>("odom_init/cloud_surf", 100);
        pubLaserCloudFullRes = this->create_publisher<sensor_msgs::msg::PointCloud2>("odom_init/cloud_full", 100);
        pubLaserOdometry = this->create_publisher<nav_msgs::msg::Odometry>("odom_init/laser_odom", 100);
        pubLaserPath = this->create_publisher<nav_msgs::msg::Path>("odom_init/path", 100);
        odom_cloud_pub = this->create_publisher<lo_sam::msg::CloudInfo>("odom_init/odom_cloud", 100);

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudFullRes.reset(new pcl::PointCloud<PointType>());

        para_q[0] = 0; para_q[1] = 0; para_q[2] = 0; para_q[3] = 1;
        para_t[0] = 0; para_t[1] = 0; para_t[2] = 0;
        q_last_curr = Eigen::Map<Eigen::Quaterniond>(para_q);
        t_last_curr = Eigen::Map<Eigen::Vector3d>(para_t);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&LaserOdometryNode::process, this));
    }

private:
    // Callback handlers just push data into buffers with mutex locking
    void laserCloudSharpHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mBuf);
        cornerSharpBuf.push(msg);
    }
    void laserCloudLessSharpHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mBuf);
        cornerLessSharpBuf.push(msg);
    }
    void laserCloudFlatHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mBuf);
        surfFlatBuf.push(msg);
    }
    void laserCloudLessFlatHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mBuf);
        surfLessFlatBuf.push(msg);
    }
    void laserCloudFullResHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(mBuf);
        fullPointsBuf.push(msg);
    }

    // undistort lidar point
    void TransformToStart(const PointType *pi, PointType *po)
    {
        double s = DISTORTION ? (pi->intensity - int(pi->intensity)) / SCAN_PERIOD : 1.0;
        Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
        Eigen::Vector3d t_point_last = s * t_last_curr;
        Eigen::Vector3d point(pi->x, pi->y, pi->z);
        Eigen::Vector3d un_point = q_point_last * point + t_point_last;

        po->x = un_point.x();
        po->y = un_point.y();
        po->z = un_point.z();
        po->intensity = pi->intensity;
    }
    void TransformToEnd(const PointType *pi, PointType *po)
    {
        PointType un_point_tmp;
        TransformToStart(pi, &un_point_tmp);

        Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
        Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

        po->x = point_end.x();
        po->y = point_end.y();
        po->z = point_end.z();
        po->intensity = int(pi->intensity);
    }

    void process()
    {
        if (!rclcpp::ok())
            return;

        std::lock_guard<std::mutex> lock(mBuf);
        if (cornerSharpBuf.empty() || cornerLessSharpBuf.empty() ||
            surfFlatBuf.empty() || surfLessFlatBuf.empty() ||
            fullPointsBuf.empty())
            return;

        auto timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp;
        auto timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp;
        auto timeSurfPointsFlat = surfFlatBuf.front()->header.stamp;
        auto timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp;
        auto timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp;

        if (timeCornerPointsSharp != timeLaserCloudFullRes ||
            timeCornerPointsLessSharp != timeLaserCloudFullRes ||
            timeSurfPointsFlat != timeLaserCloudFullRes ||
            timeSurfPointsLessFlat != timeLaserCloudFullRes)
        {
            RCLCPP_WARN(this->get_logger(), "Unsynchronized cloud info!");
            return;
        }

        pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
        cornerSharpBuf.pop();

        pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
        cornerLessSharpBuf.pop();

        pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
        surfFlatBuf.pop();

        pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
        surfLessFlatBuf.pop();

        pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
        fullPointsBuf.pop();

        // main odometry processing
        TicToc t_whole;
        if (!systemInited)
        {
            systemInited = true;
            RCLCPP_INFO(this->get_logger(), "Initialization finished");
            return;
        }
        else
        {
            int cornerPointsSharpNum = cornerPointsSharp->points.size();
            int surfPointsFlatNum = surfPointsFlat->points.size();

            // Ceres optimization
            TicToc t_opt;
            for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
            {
                corner_correspondence = 0;
                plane_correspondence = 0;

                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;
                ceres::Problem problem(problem_options);

                problem.AddParameterBlock(para_q, 4, q_parameterization);
                problem.AddParameterBlock(para_t, 3);

                pcl::PointXYZI pointSel;
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                // Find corner correspondences
                for (int i = 0; i < cornerPointsSharpNum; ++i)
                {
                    TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                    if (kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis) > 0)
                    {
                        int closestPointInd = -1, minPointInd2 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                            // Search upwards scan lines
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                if (pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                            // Search downwards scan lines
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                if (pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }

                        if (minPointInd2 >= 0)
                        {
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s = DISTORTION ?
                                (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD : 1.0;

                            ceres::CostFunction *cost_function =
                                LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                        }
                    }
                }

                // Find plane correspondences
                for (int i = 0; i < surfPointsFlatNum; ++i)
                {
                    TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                    if (kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis) > 0)
                    {
                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;
                            // Search increasing scan lines
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }
                            // Search decreasing scan lines
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }
                        }

                        if (minPointInd2 >= 0 && minPointInd3 >= 0)
                        {
                            Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                       surfPointsFlat->points[i].y,
                                                       surfPointsFlat->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                        laserCloudSurfLast->points[closestPointInd].y,
                                                        laserCloudSurfLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                        laserCloudSurfLast->points[minPointInd2].y,
                                                        laserCloudSurfLast->points[minPointInd2].z);
                            Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                        laserCloudSurfLast->points[minPointInd3].y,
                                                        laserCloudSurfLast->points[minPointInd3].z);

                            double s = DISTORTION ?
                                (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD : 1.0;

                            ceres::CostFunction *cost_function =
                                LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            plane_correspondence++;
                        }
                    }
                }

                if ((corner_correspondence + plane_correspondence) < 10)
                {
                    RCLCPP_WARN(this->get_logger(), "Less correspondence in optimization");
                }

                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.minimizer_progress_to_stdout = false;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
            }

            t_w_curr = t_w_curr + q_w_curr * t_last_curr;
            q_w_curr = q_w_curr * q_last_curr;
        }

        // Publish result
        nav_msgs::msg::Odometry laserOdometry;
        laserOdometry.header.frame_id = "camera_init";
        laserOdometry.child_frame_id = "laser_odom";
        laserOdometry.header.stamp = time_from_builtin(timeSurfPointsLessFlat);
        laserOdometry.pose.pose.orientation.x = q_w_curr.x();
        laserOdometry.pose.pose.orientation.y = q_w_curr.y();
        laserOdometry.pose.pose.orientation.z = q_w_curr.z();
        laserOdometry.pose.pose.orientation.w = q_w_curr.w();
        laserOdometry.pose.pose.position.x = t_w_curr.x();
        laserOdometry.pose.pose.position.y = t_w_curr.y();
        laserOdometry.pose.pose.position.z = t_w_curr.z();
        pubLaserOdometry->publish(laserOdometry);

        geometry_msgs::msg::PoseStamped laserPose;
        laserPose.header = laserOdometry.header;
        laserPose.pose = laserOdometry.pose.pose;
        laserPath.header.stamp = laserOdometry.header.stamp;
        laserPath.poses.push_back(laserPose);
        laserPath.header.frame_id = "camera_init";
        // pubLaserPath->publish(laserPath);

        // Swap point clouds
        auto laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        laserCloudCornerLastNum = static_cast<int>(laserCloudCornerLast->points.size());
        laserCloudSurfLastNum = static_cast<int>(laserCloudSurfLast->points.size());

        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

        // Publish point clouds
        sensor_msgs::msg::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = time_from_builtin(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "camera_init";
        pubLaserCloudCornerLast->publish(laserCloudCornerLast2);

        sensor_msgs::msg::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = time_from_builtin(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "camera_init";
        pubLaserCloudSurfLast->publish(laserCloudSurfLast2);

        sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = time_from_builtin(timeSurfPointsLessFlat);
        laserCloudFullRes3.header.frame_id = "camera_init";
        pubLaserCloudFullRes->publish(laserCloudFullRes3);

        lo_sam::msg::CloudInfo cloud_odom_msg;
        cloud_odom_msg.header = laserCloudCornerLast2.header;
        cloud_odom_msg.odom_init = laserOdometry;
        cloud_odom_msg.cloud_full = laserCloudFullRes3;
        cloud_odom_msg.cloud_surf = laserCloudSurfLast2;
        cloud_odom_msg.cloud_edge = laserCloudCornerLast2;
        odom_cloud_pub->publish(cloud_odom_msg);

        if (t_whole.toc() > 100)
        {
            RCLCPP_WARN(this->get_logger(), "Odometry process over 100ms");
        }

        frameCount++;
    }

    rclcpp::Time time_from_builtin(const builtin_interfaces::msg::Time & t)
    {
        return rclcpp::Time(t.sec, t.nanosec);
    }

private:
    int corner_correspondence, plane_correspondence;
    const double SCAN_PERIOD;
    const double DISTANCE_SQ_THRESHOLD;
    const double NEARBY_SCAN;

    int skipFrameNum;
    bool systemInited;

    double timeCornerPointsSharp;
    double timeCornerPointsLessSharp;
    double timeSurfPointsFlat;
    double timeSurfPointsLessFlat;
    double timeLaserCloudFullRes;

    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudFullRes;

    int laserCloudCornerLastNum;
    int laserCloudSurfLastNum;

    Eigen::Quaterniond q_w_curr = Eigen::Quaterniond(1, 0, 0, 0);
    Eigen::Vector3d t_w_curr = Eigen::Vector3d(0, 0, 0);

    double para_q[4];
    double para_t[3];
    Eigen::Map<Eigen::Quaterniond> q_last_curr{para_q};
    Eigen::Map<Eigen::Vector3d> t_last_curr{para_t};

    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> cornerSharpBuf;
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> cornerLessSharpBuf;
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> surfFlatBuf;
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> surfLessFlatBuf;
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> fullPointsBuf;

    std::mutex mBuf;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subCornerPointsSharp;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subCornerPointsLessSharp;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subSurfPointsFlat;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subSurfPointsLessFlat;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloudFullRes;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudCornerLast;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurfLast;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubLaserPath;
    rclcpp::Publisher<lo_sam::msg::CloudInfo>::SharedPtr odom_cloud_pub;

    nav_msgs::msg::Path laserPath;

    int frameCount;

    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LaserOdometryNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
