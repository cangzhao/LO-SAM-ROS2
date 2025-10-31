#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <opencv2/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <chrono>

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType
{
    VELODYNE,
    OUSTER
};

class ParamServer
{
public:
    std::shared_ptr<rclcpp::Node> node;

    std::string robot_id;

    // Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;

    // Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Lidar Sensor Configuration
    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter params
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize;

    float z_tolerance;
    float rotation_tolerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;
    float surroundingkeyframeAddingAngleThreshold;
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    // Loop closure
    bool loopClosureEnableFlag;
    float loopClosureFrequency;
    int surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    explicit ParamServer(std::shared_ptr<rclcpp::Node> node_)
        : node(node_)
    {
        robot_id = node->declare_parameter<std::string>("robot_id", "roboat");

        pointCloudTopic = node->declare_parameter<std::string>("lio_sam.pointCloudTopic", "points_raw");
        imuTopic = node->declare_parameter<std::string>("lio_sam.imuTopic", "imu_correct");
        odomTopic = node->declare_parameter<std::string>("lio_sam.odomTopic", "odometry/imu");
        gpsTopic = node->declare_parameter<std::string>("lio_sam.gpsTopic", "odometry/gps");

        lidarFrame = node->declare_parameter<std::string>("lio_sam.lidarFrame", "base_link");
        baselinkFrame = node->declare_parameter<std::string>("lio_sam.baselinkFrame", "base_link");
        odometryFrame = node->declare_parameter<std::string>("lio_sam.odometryFrame", "odom");
        mapFrame = node->declare_parameter<std::string>("lio_sam.mapFrame", "map");

        useImuHeadingInitialization = node->declare_parameter<bool>("lio_sam.useImuHeadingInitialization", false);
        useGpsElevation = node->declare_parameter<bool>("lio_sam.useGpsElevation", false);
        gpsCovThreshold = node->declare_parameter<float>("lio_sam.gpsCovThreshold", 2.0f);
        poseCovThreshold = node->declare_parameter<float>("lio_sam.poseCovThreshold", 25.0f);

        savePCD = node->declare_parameter<bool>("lio_sam.savePCD", false);
        savePCDDirectory = node->declare_parameter<std::string>("lio_sam.savePCDDirectory", "/Downloads/LOAM/");

        std::string sensorStr = node->declare_parameter<std::string>("lio_sam.sensor", "");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else
        {
            RCLCPP_ERROR(node->get_logger(),
                "Invalid sensor type (must be either 'velodyne' or 'ouster'): %s", sensorStr.c_str());
            rclcpp::shutdown();
        }

        N_SCAN = node->declare_parameter<int>("lio_sam.N_SCAN", 16);
        Horizon_SCAN = node->declare_parameter<int>("lio_sam.Horizon_SCAN", 1800);
        downsampleRate = node->declare_parameter<int>("lio_sam.downsampleRate", 1);
        lidarMinRange = node->declare_parameter<float>("lio_sam.lidarMinRange", 1.0f);
        lidarMaxRange = node->declare_parameter<float>("lio_sam.lidarMaxRange", 1000.0f);

        imuAccNoise = node->declare_parameter<float>("lio_sam.imuAccNoise", 0.01f);
        imuGyrNoise = node->declare_parameter<float>("lio_sam.imuGyrNoise", 0.001f);
        imuAccBiasN = node->declare_parameter<float>("lio_sam.imuAccBiasN", 0.0002f);
        imuGyrBiasN = node->declare_parameter<float>("lio_sam.imuGyrBiasN", 0.00003f);
        imuGravity = node->declare_parameter<float>("lio_sam.imuGravity", 9.80511f);
        imuRPYWeight = node->declare_parameter<float>("lio_sam.imuRPYWeight", 0.01f);

        extRotV = node->declare_parameter<std::vector<double>>("lio_sam.extrinsicRot", std::vector<double>{});
        extRPYV = node->declare_parameter<std::vector<double>>("lio_sam.extrinsicRPY", std::vector<double>{});
        extTransV = node->declare_parameter<std::vector<double>>("lio_sam.extrinsicTrans", std::vector<double>{});
        if (extRotV.size() == 9)
            extRot = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(extRotV.data());
        if (extRPYV.size() == 9)
            extRPY = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(extRPYV.data());
        if (extTransV.size() == 3)
            extTrans = Eigen::Map<const Eigen::Matrix<double, 3, 1, Eigen::RowMajor>>(extTransV.data());
        extQRPY = Eigen::Quaterniond(extRPY);

        edgeThreshold = node->declare_parameter<float>("lio_sam.edgeThreshold", 0.1f);
        surfThreshold = node->declare_parameter<float>("lio_sam.surfThreshold", 0.1f);
        edgeFeatureMinValidNum = node->declare_parameter<int>("lio_sam.edgeFeatureMinValidNum", 10);
        surfFeatureMinValidNum = node->declare_parameter<int>("lio_sam.surfFeatureMinValidNum", 100);

        odometrySurfLeafSize = node->declare_parameter<float>("lio_sam.odometrySurfLeafSize", 0.2f);
        mappingCornerLeafSize = node->declare_parameter<float>("lio_sam.mappingCornerLeafSize", 0.2f);
        mappingSurfLeafSize = node->declare_parameter<float>("lio_sam.mappingSurfLeafSize", 0.2f);

        z_tolerance = node->declare_parameter<float>("lio_sam.z_tolerance", FLT_MAX);
        rotation_tolerance = node->declare_parameter<float>("lio_sam.rotation_tolerance", FLT_MAX);

        numberOfCores = node->declare_parameter<int>("lio_sam.numberOfCores", 2);
        mappingProcessInterval = node->declare_parameter<double>("lio_sam.mappingProcessInterval", 0.15);

        surroundingkeyframeAddingDistThreshold = node->declare_parameter<float>("lio_sam.surroundingkeyframeAddingDistThreshold", 1.0f);
        surroundingkeyframeAddingAngleThreshold = node->declare_parameter<float>("lio_sam.surroundingkeyframeAddingAngleThreshold", 0.2f);
        surroundingKeyframeDensity = node->declare_parameter<float>("lio_sam.surroundingKeyframeDensity", 1.0f);
        surroundingKeyframeSearchRadius = node->declare_parameter<float>("lio_sam.surroundingKeyframeSearchRadius", 50.0f);

        loopClosureEnableFlag = node->declare_parameter<bool>("lio_sam.loopClosureEnableFlag", false);
        loopClosureFrequency = node->declare_parameter<float>("lio_sam.loopClosureFrequency", 1.0f);
        surroundingKeyframeSize = node->declare_parameter<int>("lio_sam.surroundingKeyframeSize", 50);
        historyKeyframeSearchRadius = node->declare_parameter<float>("lio_sam.historyKeyframeSearchRadius", 10.0f);
        historyKeyframeSearchTimeDiff = node->declare_parameter<float>("lio_sam.historyKeyframeSearchTimeDiff", 30.0f);
        historyKeyframeSearchNum = node->declare_parameter<int>("lio_sam.historyKeyframeSearchNum", 25);
        historyKeyframeFitnessScore = node->declare_parameter<float>("lio_sam.historyKeyframeFitnessScore", 0.3f);

        globalMapVisualizationSearchRadius = node->declare_parameter<float>("lio_sam.globalMapVisualizationSearchRadius", 1e3f);
        globalMapVisualizationPoseDensity = node->declare_parameter<float>("lio_sam.globalMapVisualizationPoseDensity", 10.0f);
        globalMapVisualizationLeafSize = node->declare_parameter<float>("lio_sam.globalMapVisualizationLeafSize", 1.0f);
    }

    sensor_msgs::msg::Imu imuConverter(const sensor_msgs::msg::Imu &imu_in)
    {
        sensor_msgs::msg::Imu imu_out = imu_in;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();

        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();

        // 如果需要旋转姿态，可以用 tf2 相关函数进行修改
        /*
        geometry_msgs::msg::Quaternion q = imu_in.orientation;
        tf2::Quaternion tf2_q;
        tf2::fromMsg(q, tf2_q);
        tf2::Quaternion q_final = tf2_q * tf2::Quaternion(extQRPY.x(), extQRPY.y(), extQRPY.z(), extQRPY.w());
        imu_out.orientation = tf2::toMsg(q_final);

        double norm = q_final.length();
        if (norm < 0.1)
        {
            RCLCPP_ERROR(node->get_logger(), "Invalid quaternion, please use a 9-axis IMU!");
            rclcpp::shutdown();
        }
        */

        return imu_out;
    }
};

inline sensor_msgs::msg::PointCloud2 publishCloud(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub,
    pcl::PointCloud<PointType>::Ptr thisCloud,
    rclcpp::Time thisStamp,
    const std::string & thisFrame)
{
    sensor_msgs::msg::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->get_subscription_count() > 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

template <typename T>
double ROS_TIME(const T & msg)
{
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
}

template <typename T>
void imuAngular2rosAngular(const sensor_msgs::msg::Imu & imuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = imuMsg.angular_velocity.x;
    *angular_y = imuMsg.angular_velocity.y;
    *angular_z = imuMsg.angular_velocity.z;
}

template <typename T>
void imuAccel2rosAccel(const sensor_msgs::msg::Imu & imuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = imuMsg.linear_acceleration.x;
    *acc_y = imuMsg.linear_acceleration.y;
    *acc_z = imuMsg.linear_acceleration.z;
}

template <typename T>
void imuRPY2rosRPY(const sensor_msgs::msg::Imu & imuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf2::Quaternion orientation;
    tf2::fromMsg(imuMsg.orientation, orientation);
    tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}

inline float pointDistance(PointType p)
{
    return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

inline float pointDistance(PointType p1, PointType p2)
{
    return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                     (p1.y - p2.y) * (p1.y - p2.y) +
                     (p1.z - p2.z) * (p1.z - p2.z));
}

class tictoc
{
public:
    tictoc()
    {
        start = std::chrono::system_clock::now();
    }
    void Tic()
    {
        start = std::chrono::system_clock::now();
    }
    void Toc()
    {
        end = std::chrono::system_clock::now();
        times = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    int64_t times;

private:
    std::chrono::system_clock::time_point start, end;
};

#endif
