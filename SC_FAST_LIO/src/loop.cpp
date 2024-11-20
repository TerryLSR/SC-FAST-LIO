#include <ros/ros.h>
#include <tf/tf.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>
#include <nav_msgs/Odometry.h>

#include <sensor_msgs/PointCloud2.h>
#include <thread>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "Scancontext.h"



class LoopClosure
{
    typedef struct
    {
        double pos_x;
        double pos_y;
        double pos_z;

        double ori_x;
        double ori_y;
        double ori_z;
        double ori_w;

        double roll;
        double pitch;
        double yaw;

        double time;
        bool has_cloud_paired = false;
    }CloudPoseXYZOri;



private:
    ros::NodeHandle nh_;
    // ros::Subscriber odom_sub_;
    // ros::Subscriber key_frame_cloud_sub_;

    ros::Publisher key_frame_pose_pub_;
    ros::Publisher pubLoopConstraintEdge_;
    ros::Publisher pubIcpKeyFrames_;
    ros::Publisher pubGlobalMap_;

    std::string front_laser_odom_topic_ = "/Odometry";
    std::string cloud_topic_ = "/cloud_registered_body";
    bool first_frame_ = true;

    std::deque<CloudPoseXYZOri> key_frame_odom_poses_, corrected_frame_pose_;

    typedef pcl::PointXYZI  PointType;
    pcl::PointCloud<PointType>::Ptr path_cloud_;
    pcl::PointCloud<PointType>::Ptr corrected_path_cloud_;

    std::vector<std::pair<int, int>> loopIndexQueue;
    std::vector<gtsam::Pose3> loopPoseQueue;
    std::vector<gtsam::SharedNoiseModel> loopNoiseQueue; // giseop for polymorhpisam (Diagonal <- Gausssian <- Base)
    typedef struct _LOOP_IDX_
    {
        int cur_idx;
        int history_idx;
        bool operator < (const _LOOP_IDX_ &o) const
        {
            return cur_idx < o.cur_idx || history_idx < o.history_idx;
        }
    }LOOPIDX;
    std::multimap<LOOPIDX, int> loopIndexContainer;

    typedef struct
    {
        pcl::PointCloud<PointType>::Ptr pc_ptr;
        double pc_time;
    }LidarData;
    
    std::deque<LidarData> lidar_buffer_ ,all_lidar_buffer_;
    std::mutex data_lock_;

    message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> key_frame_cloud_sub_;

    typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::PointCloud2> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

    // loop detector 
    SCManager scManager_;
    int historyKeyframeSearchNum_ = 25;
    pcl::PointCloud<PointType>::Ptr SClatestSurfKeyFrameCloud_;
    pcl::PointCloud<PointType>::Ptr SCnearHistorySurfKeyFrameCloud_;
    pcl::PointCloud<PointType>::Ptr SCnearHistorySurfKeyFrameCloudDS_;
    pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames_;
    pcl::VoxelGrid<PointType> downSizeFilterKeyFrame_;
    pcl::VoxelGrid<PointType> downSizeGlobalMap_;

    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS_;

    double historyKeyframeFitnessScore_ = 0.16;
    int SCclosestHistoryFrameID_;

    unsigned short loop_frame_skip_cnt_ = 0;

    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::Values initialEstimate;
    gtsam::ISAM2 *isam;
    gtsam::Values isamCurrentEstimate;

    gtsam::noiseModel::Diagonal::shared_ptr priorNoise;
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise;

public:
    LoopClosure(/* args */){
        // key_frame_cloud_sub_ = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered_body", 2, &LoopClosure::cloud_callbac   k, this);
        // odom_sub_ = nh_.subscribe(front_laser_odom_topic_, 200000, &LoopClosure::odom_callback, this);
        key_frame_pose_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/keyframe_pose", 100000, true);
        pubIcpKeyFrames_ = nh_.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2, true);
        pubGlobalMap_ = nh_.advertise<sensor_msgs::PointCloud2>("/global_map_cloud", 2, true);

        odom_sub_.subscribe(nh_, front_laser_odom_topic_, 1);
        key_frame_cloud_sub_.subscribe(nh_, cloud_topic_, 1);
        sync_.reset(new Sync(MySyncPolicy(5), odom_sub_, key_frame_cloud_sub_));
        sync_->registerCallback(boost::bind(&LoopClosure::SyncDataCallback, this, _1, _2));

        pubLoopConstraintEdge_ = nh_.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 1, true);
        
        path_cloud_.reset(new pcl::PointCloud<PointType>());
        corrected_path_cloud_.reset(new pcl::PointCloud<PointType>());

        SClatestSurfKeyFrameCloud_.reset(new pcl::PointCloud<PointType>());
        SCnearHistorySurfKeyFrameCloud_.reset(new pcl::PointCloud<PointType>());
        SCnearHistorySurfKeyFrameCloudDS_.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFramesDS_.reset(new pcl::PointCloud<PointType>());
        
        double filter_size = 0.3; downSizeFilterHistoryKeyFrames_.setLeafSize(filter_size, filter_size, filter_size);
        filter_size = 0.2; downSizeFilterKeyFrame_.setLeafSize(filter_size, filter_size, filter_size);
        filter_size = 0.5;downSizeGlobalMap_.setLeafSize(filter_size, filter_size, filter_size);
        
        gtsam::ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.01;
		parameters.relinearizeSkip = 1;
    	isam = new gtsam::ISAM2(parameters);

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
        priorNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
        odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

        std::thread loopthread(&LoopClosure::loopClosureThread, this);
        loopthread.detach();
        std::thread visualizeMapThread(&LoopClosure::visualizeGlobalMapThread, this);
        visualizeMapThread.detach();
    }
    
    void SyncDataCallback(const nav_msgs::OdometryConstPtr &odom_in, const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
    {
        if(first_frame_ == true)
        {
            first_frame_ = false;

            CloudPoseXYZOri this_cloud_pose;
            this_cloud_pose.time = odom_in->header.stamp.toSec();
            this_cloud_pose.ori_x = odom_in->pose.pose.orientation.x;
            this_cloud_pose.ori_y = odom_in->pose.pose.orientation.y;
            this_cloud_pose.ori_z = odom_in->pose.pose.orientation.z;
            this_cloud_pose.ori_w = odom_in->pose.pose.orientation.w;

            this_cloud_pose.pos_x = odom_in->pose.pose.position.x;
            this_cloud_pose.pos_y = odom_in->pose.pose.position.y;
            this_cloud_pose.pos_z = odom_in->pose.pose.position.z;
            
            tf::Quaternion tf_q(odom_in->pose.pose.orientation.x, odom_in->pose.pose.orientation.y, 
                        odom_in->pose.pose.orientation.z, odom_in->pose.pose.orientation.w );//x y z w
            double roll, pitch, yaw_cur;//定义存储r\p\y的容器
            tf::Matrix3x3(tf_q).getRPY(roll, pitch, yaw_cur);
            this_cloud_pose.roll = roll; this_cloud_pose.pitch = pitch; this_cloud_pose.yaw = yaw_cur;

            key_frame_odom_poses_.push_back(this_cloud_pose);

            PointType key_frame_point;
            key_frame_point.x = odom_in->pose.pose.position.x;
            key_frame_point.y = odom_in->pose.pose.position.y;
            key_frame_point.z = odom_in->pose.pose.position.z;
            key_frame_point.intensity = odom_in->pose.pose.position.z;
            path_cloud_->points.push_back(key_frame_point);

            // PubKeyFramePose(path_cloud_, this_cloud_pose.time);

            pcl::PointCloud<PointType>::Ptr pl_orig(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*cloud_msg, *pl_orig);

            lidar_buffer_.push_back(LidarData{pl_orig, cloud_msg->header.stamp.toSec()});

            return;
        }

        data_lock_.lock();

        CloudPoseXYZOri this_cloud_pose;
        this_cloud_pose.time = odom_in->header.stamp.toSec();
        this_cloud_pose.ori_x = odom_in->pose.pose.orientation.x;
        this_cloud_pose.ori_y = odom_in->pose.pose.orientation.y;
        this_cloud_pose.ori_z = odom_in->pose.pose.orientation.z;
        this_cloud_pose.ori_w = odom_in->pose.pose.orientation.w;

        this_cloud_pose.pos_x = odom_in->pose.pose.position.x;
        this_cloud_pose.pos_y = odom_in->pose.pose.position.y;
        this_cloud_pose.pos_z = odom_in->pose.pose.position.z;

        if(DeltaDisSquare(this_cloud_pose, key_frame_odom_poses_.back()) > 0.3*0.3)
        {   

            tf::Quaternion tf_q(odom_in->pose.pose.orientation.x, odom_in->pose.pose.orientation.y, 
                        odom_in->pose.pose.orientation.z, odom_in->pose.pose.orientation.w );//x y z w
            double roll, pitch, yaw_cur;//定义存储r\p\y的容器
            tf::Matrix3x3(tf_q).getRPY(roll, pitch, yaw_cur);
            this_cloud_pose.roll = roll; this_cloud_pose.pitch = pitch; this_cloud_pose.yaw = yaw_cur;
            key_frame_odom_poses_.push_back(this_cloud_pose);

            PointType key_frame_point;
            key_frame_point.x = odom_in->pose.pose.position.x;
            key_frame_point.y = odom_in->pose.pose.position.y;
            key_frame_point.z = odom_in->pose.pose.position.z;
            key_frame_point.intensity = odom_in->pose.pose.position.z;
            path_cloud_->points.push_back(key_frame_point);

            // PubKeyFramePose(path_cloud_, this_cloud_pose.time);

            pcl::PointCloud<PointType>::Ptr pl_orig(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*cloud_msg, *pl_orig);

            lidar_buffer_.push_back(LidarData{pl_orig, cloud_msg->header.stamp.toSec()});
        }
        data_lock_.unlock();

    }

    double DeltaDisSquare(CloudPoseXYZOri& pose1, CloudPoseXYZOri& pose2)
    {
        return pow(pose1.pos_x - pose2.pos_x, 2) + pow(pose1.pos_y - pose2.pos_y, 2) + pow(pose1.pos_z - pose2.pos_z, 2);
    }

    void PubKeyFramePose(const pcl::PointCloud<PointType>::Ptr& path_in, double time)
    {
        sensor_msgs::PointCloud2 PoseCloudMap;
        pcl::toROSMsg(*path_in, PoseCloudMap);
        PoseCloudMap.header.stamp = ros::Time().fromSec(time);
        PoseCloudMap.header.frame_id = "camera_init";
        key_frame_pose_pub_.publish(PoseCloudMap);
    }

    void visualizeLoopClosure()
    {
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = "camera_init";
        markerNode.header.stamp = ros::Time::now();
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = "camera_init";
        markerEdge.header.stamp = ros::Time::now();
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1; markerEdge.scale.y = 0.1; markerEdge.scale.z = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first.cur_idx;
            int key_pre = it->first.history_idx;
            geometry_msgs::Point p;
            p.x = corrected_path_cloud_->points[key_cur].x;
            p.y = corrected_path_cloud_->points[key_cur].y;
            p.z = corrected_path_cloud_->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = corrected_path_cloud_->points[key_pre].x;
            p.y = corrected_path_cloud_->points[key_pre].y;
            p.z = corrected_path_cloud_->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge_.publish(markerArray);
    }

    void visualizeGlobalMapThread(){
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            // publishGlobalMap();
        }       
    }

    //TODO: publishGlobalMap another thread 
    void publishGlobalMap(){
        
        if (pubGlobalMap_.getNumSubscribers() == 0)
            return;
        globalMapKeyFramesDS_->clear();
        pcl::PointCloud<PointType>::Ptr tmp_cloud(new pcl::PointCloud<PointType>());
        int lidar_size = corrected_frame_pose_.size();
        Eigen::Matrix4d tmp_pose;
        for (int i = 0; i < lidar_size; ++i){
			tmp_pose = KeyFramePose2EigenPose(corrected_frame_pose_[i]);
            pcl::transformPointCloud(*all_lidar_buffer_[i].pc_ptr, *tmp_cloud, tmp_pose);
			*globalMapKeyFramesDS_ += *tmp_cloud;
        }
	    // downsample visualized points
        downSizeGlobalMap_.setInputCloud(globalMapKeyFramesDS_);
        downSizeGlobalMap_.filter(*globalMapKeyFramesDS_);

        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*globalMapKeyFramesDS_, cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time::now(); // use sim time
        cloudMsgTemp.header.frame_id = "camera_init";
        pubGlobalMap_.publish(cloudMsgTemp);  
        
    }

    void loopClosureThread(){

        ros::Rate rate(10); // lidar rate
        while (ros::ok()){
            rate.sleep();
            performLoopClosure();
            
            visualizeLoopClosure();
        }

    } // loopClosureThread

    Eigen::Matrix4d KeyFramePose2EigenPose(const CloudPoseXYZOri& key_frame_pose)
    {
        Eigen::Matrix4d eigen_pose = Eigen::Matrix4d::Identity();
        eigen_pose(0, 3) = key_frame_pose.pos_x;
        eigen_pose(1, 3) = key_frame_pose.pos_y;
        eigen_pose(2, 3) = key_frame_pose.pos_z;
        Eigen::Quaterniond quaternion(key_frame_pose.ori_w, key_frame_pose.ori_x,
                                        key_frame_pose.ori_y, key_frame_pose.ori_z);
        eigen_pose.block<3, 3>(0, 0) = quaternion.matrix();

        return eigen_pose;
    }

    bool detectLoopClosure(const int loop_idx){
        
        SClatestSurfKeyFrameCloud_->clear();
        SCnearHistorySurfKeyFrameCloud_->clear();
        SCnearHistorySurfKeyFrameCloudDS_->clear();

        int latestFrameIDLoopCloure = loop_idx;
        SCclosestHistoryFrameID_ = -1; // init with -1
        auto detectResult = scManager_.detectLoopClosureID(); // first: nn index, second: yaw diff 
        SCclosestHistoryFrameID_ = detectResult.first;
        double yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)

        // if all close, reject
        if (SCclosestHistoryFrameID_ == -1){ 
            return false;
        }

        if(fabs(SCclosestHistoryFrameID_ - latestFrameIDLoopCloure) < 10)
        {
            std::cout << "this key is too close.not add loop " << latestFrameIDLoopCloure << ",history indx " << SCclosestHistoryFrameID_ << std::endl;
            return false;
        }

        LOOPIDX loop_idx_pair{latestFrameIDLoopCloure, SCclosestHistoryFrameID_};
        auto it = loopIndexContainer.find(loop_idx_pair);
        if (it != loopIndexContainer.end())
        {
            std::cout << "this key has looped " << latestFrameIDLoopCloure << ",history indx " << SCclosestHistoryFrameID_ << std::endl;
            return false;
        }

        // save latest key frames: query ptcloud (corner points + surface points)
        // NOTE: using "closestHistoryFrameID" to make same root of submap points to get a direct relative between the query point cloud (latestSurfKeyFrameCloud) and the map (nearHistorySurfKeyFrameCloud). by giseop
        // i.e., set the query point cloud within mapside's local coordinate
        Eigen::Matrix4d sc_last_pose = KeyFramePose2EigenPose(corrected_frame_pose_[SCclosestHistoryFrameID_]);
        pcl::transformPointCloud(*all_lidar_buffer_[loop_idx].pc_ptr, *SClatestSurfKeyFrameCloud_, sc_last_pose);

	    // save history near key frames: map ptcloud (icp to query ptcloud)
        Eigen::Matrix4d sc_his_pose;
        for (int j = -historyKeyframeSearchNum_; j <= historyKeyframeSearchNum_; ++j){
            if (SCclosestHistoryFrameID_ + j < 0 || SCclosestHistoryFrameID_ + j > latestFrameIDLoopCloure)
                continue;
            sc_his_pose = KeyFramePose2EigenPose(corrected_frame_pose_[SCclosestHistoryFrameID_+j]);
            pcl::transformPointCloud(*all_lidar_buffer_[SCclosestHistoryFrameID_+j].pc_ptr, *SCnearHistorySurfKeyFrameCloud_, sc_his_pose);
            *SCnearHistorySurfKeyFrameCloudDS_ += *SCnearHistorySurfKeyFrameCloud_;
        }
        downSizeFilterHistoryKeyFrames_.setInputCloud(SCnearHistorySurfKeyFrameCloudDS_);
        downSizeFilterHistoryKeyFrames_.filter(*SCnearHistorySurfKeyFrameCloudDS_);

        return true;
    }

    void performLoopClosure()
    {
        data_lock_.lock();
        std::deque<LidarData> tmp_lidar_buffer_ = lidar_buffer_; // tmp_lidar_buffer_ need find loop
        lidar_buffer_.clear();
        std::deque<CloudPoseXYZOri> tmp_key_frame_odom_pose_ = key_frame_odom_poses_;
        data_lock_.unlock();

        if(tmp_lidar_buffer_.empty())
            return;

        int lidar_idx = 0;
        int lidar_size_before_add = all_lidar_buffer_.size();
        all_lidar_buffer_.insert(all_lidar_buffer_.end(), tmp_lidar_buffer_.begin(), tmp_lidar_buffer_.end());
        
        //find loop closure
        while(!tmp_lidar_buffer_.empty())
        {
            LidarData tmp_data = tmp_lidar_buffer_.front();
            tmp_lidar_buffer_.pop_front();
            lidar_idx ++;
            int loop_idx = lidar_size_before_add - 1 + lidar_idx;
            
            //add isam factor
            if(loop_idx == 0)
            {
                gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, 
                                    gtsam::Pose3(gtsam::Rot3::RzRyRx(tmp_key_frame_odom_pose_[loop_idx].roll, tmp_key_frame_odom_pose_[loop_idx].pitch, tmp_key_frame_odom_pose_[loop_idx].yaw),
                                                gtsam::Point3(tmp_key_frame_odom_pose_[loop_idx].pos_x, tmp_key_frame_odom_pose_[loop_idx].pos_y, tmp_key_frame_odom_pose_[loop_idx].pos_z)), priorNoise));
                initialEstimate.insert(0, gtsam::Pose3(gtsam::Rot3::RzRyRx(tmp_key_frame_odom_pose_[loop_idx].roll, tmp_key_frame_odom_pose_[loop_idx].pitch, tmp_key_frame_odom_pose_[loop_idx].yaw),
                                                  gtsam::Point3(tmp_key_frame_odom_pose_[loop_idx].pos_x, tmp_key_frame_odom_pose_[loop_idx].pos_y, tmp_key_frame_odom_pose_[loop_idx].pos_z)));
            }
            else
            {
                // add delta use origin odom
                gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(tmp_key_frame_odom_pose_[loop_idx-1].roll, tmp_key_frame_odom_pose_[loop_idx-1].pitch, tmp_key_frame_odom_pose_[loop_idx-1].yaw),
                                                gtsam::Point3(tmp_key_frame_odom_pose_[loop_idx-1].pos_x, tmp_key_frame_odom_pose_[loop_idx-1].pos_y, tmp_key_frame_odom_pose_[loop_idx-1].pos_z));

                gtsam::Pose3 poseTo   = gtsam::Pose3(gtsam::Rot3::RzRyRx(tmp_key_frame_odom_pose_[loop_idx].roll, tmp_key_frame_odom_pose_[loop_idx].pitch, tmp_key_frame_odom_pose_[loop_idx].yaw),
                                                    gtsam::Point3(tmp_key_frame_odom_pose_[loop_idx].pos_x, tmp_key_frame_odom_pose_[loop_idx].pos_y, tmp_key_frame_odom_pose_[loop_idx].pos_z));
                std::cout << "pose to " << poseFrom.between(poseTo) << std::endl;
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(loop_idx-1, loop_idx, poseFrom.between(poseTo), odometryNoise));
                initialEstimate.insert(loop_idx, gtsam::Pose3(gtsam::Rot3::RzRyRx(tmp_key_frame_odom_pose_[loop_idx].roll, tmp_key_frame_odom_pose_[loop_idx].pitch, tmp_key_frame_odom_pose_[loop_idx].yaw),
                                                    gtsam::Point3(tmp_key_frame_odom_pose_[loop_idx].pos_x, tmp_key_frame_odom_pose_[loop_idx].pos_y, tmp_key_frame_odom_pose_[loop_idx].pos_z)));
            }

            bool aLoopIsClosed = false;
            //add loop factor
            for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
            {
                aLoopIsClosed = true;
                int indexFrom = loopIndexQueue[i].first;
                int indexTo = loopIndexQueue[i].second;
                gtsam::Pose3 poseBetween = loopPoseQueue[i];
                // gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i]; // original 
                auto noiseBetween = loopNoiseQueue[i]; // giseop for polymorhpism // shared_ptr<gtsam::noiseModel::Base>, typedef noiseModel::Base::shared_ptr gtsam::SharedNoiseModel
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
                std::cout << "add loop " << std::endl;
            }

            loopIndexQueue.clear();
            loopPoseQueue.clear();
            loopNoiseQueue.clear();
            
            isam->update(gtSAMgraph, initialEstimate);
            isam->update();
            //update
            if (aLoopIsClosed == true)
            {
                isam->update();
                isam->update();
                isam->update();
                isam->update();
                isam->update();
            }

            gtSAMgraph.resize(0);
            initialEstimate.clear();
            
            //save the lastest corrected pose
            gtsam::Pose3 latestEstimate;
            CloudPoseXYZOri tmp_pose_data;

            isamCurrentEstimate = isam->calculateEstimate();
            latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size()-1);

            tmp_pose_data.pos_x = latestEstimate.translation().x();
            tmp_pose_data.pos_y = latestEstimate.translation().y();
            tmp_pose_data.pos_z = latestEstimate.translation().z();
            tmp_pose_data.roll  = latestEstimate.rotation().roll();
            tmp_pose_data.pitch = latestEstimate.rotation().pitch();
            tmp_pose_data.yaw   = latestEstimate.rotation().yaw();
            Eigen::Quaterniond quaternion3;
            quaternion3 = Eigen::AngleAxisd(tmp_pose_data.yaw, Eigen::Vector3d::UnitZ()) * 
                            Eigen::AngleAxisd(tmp_pose_data.pitch, Eigen::Vector3d::UnitY()) * 
                            Eigen::AngleAxisd(tmp_pose_data.roll, Eigen::Vector3d::UnitX());
            tmp_pose_data.ori_x = quaternion3.x();
            tmp_pose_data.ori_y = quaternion3.y();
            tmp_pose_data.ori_z = quaternion3.z();
            tmp_pose_data.ori_w = quaternion3.w();
            corrected_frame_pose_.push_back(tmp_pose_data);

            //for visualize
            PointType correct_point;
            correct_point.x = latestEstimate.translation().x();
            correct_point.y = latestEstimate.translation().y();
            correct_point.z = latestEstimate.translation().z();
            correct_point.intensity = latestEstimate.translation().z();
            corrected_path_cloud_->points.push_back(correct_point);

            //update all pose if loop
            if (aLoopIsClosed == true)
            {
                isamCurrentEstimate = isam->calculateEstimate();
                int numPoses = isamCurrentEstimate.size();
                
                for (int i = 0; i < numPoses; ++i){

                    tmp_pose_data.pos_x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
                    tmp_pose_data.pos_y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
                    tmp_pose_data.pos_z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();
                    tmp_pose_data.roll  = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
                    tmp_pose_data.pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
                    tmp_pose_data.yaw   = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

                    corrected_path_cloud_->points[i].x = tmp_pose_data.pos_x;
                    corrected_path_cloud_->points[i].y = tmp_pose_data.pos_y;
                    corrected_path_cloud_->points[i].z = tmp_pose_data.pos_z;
                    corrected_path_cloud_->points[i].intensity = tmp_pose_data.pos_z;

                    Eigen::Quaterniond quaternion3;
                    quaternion3 = Eigen::AngleAxisd(tmp_pose_data.yaw, Eigen::Vector3d::UnitZ()) * 
                                    Eigen::AngleAxisd(tmp_pose_data.pitch, Eigen::Vector3d::UnitY()) * 
                                    Eigen::AngleAxisd(tmp_pose_data.roll, Eigen::Vector3d::UnitX());
                    tmp_pose_data.ori_x = quaternion3.x();
                    tmp_pose_data.ori_y = quaternion3.y();
                    tmp_pose_data.ori_z = quaternion3.z();
                    tmp_pose_data.ori_w = quaternion3.w();
                    
                    corrected_frame_pose_[i] = tmp_pose_data;
                }

                publishGlobalMap();
            }

            //add to sc manager
            scManager_.makeAndSaveScancontextAndKeys(*tmp_data.pc_ptr);
            
            //downsample 
            downSizeFilterHistoryKeyFrames_.setInputCloud(all_lidar_buffer_[loop_idx].pc_ptr);
            downSizeFilterHistoryKeyFrames_.filter(*all_lidar_buffer_[loop_idx].pc_ptr);

            loop_frame_skip_cnt_ ++; // loop detect per 2 frame

            if(loop_frame_skip_cnt_ % 2 == 0)
            {
                loop_frame_skip_cnt_ = 0;

                if(corrected_frame_pose_.size() < 10) // key frame num is too small, not loop
                    continue;

                if (detectLoopClosure(loop_idx) == true)
                {
                    float x, y, z, roll, pitch, yaw;
                    Eigen::Affine3f correctionCameraFrame;
                    bool isValidSCloopFactor;
                    gtsam::Vector Vector6(6);
                    pcl::IterativeClosestPoint<PointType, PointType> icp;
                    icp.setMaxCorrespondenceDistance(100);
                    icp.setMaximumIterations(100);
                    icp.setTransformationEpsilon(1e-6);
                    icp.setEuclideanFitnessEpsilon(1e-6);
                    icp.setRANSACIterations(0);

                    // Align clouds
                    // Eigen::Affine3f icpInitialMatFoo = pcl::getTransformation(0, 0, 0, yawDiffRad, 0, 0); // because within cam coord: (z, x, y, yaw, roll, pitch)
                    // Eigen::Matrix4f icpInitialMat = icpInitialMatFoo.matrix();
                    icp.setInputSource(SClatestSurfKeyFrameCloud_);
                    icp.setInputTarget(SCnearHistorySurfKeyFrameCloudDS_);
                    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
                    icp.align(*unused_result); 
                    // icp.align(*unused_result, icpInitialMat); // PCL icp non-eye initial is bad ... don't use (LeGO LOAM author also said pcl transform is weird.)

                    std::cout << "[SC] ICP fit score: " << icp.getFitnessScore() << std::endl;
                    if ( icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore_ ) {
                        std::cout << "[SC] Reject this loop (bad icp fit score, > " << historyKeyframeFitnessScore_ << ")" << std::endl;
                        isValidSCloopFactor = false;
                    }
                    else {
                        std::cout << "[SC] The detected loop factor is added between Current [ " << loop_idx << " ] and SC nearest [ " << SCclosestHistoryFrameID_ << " ]" << std::endl;
                        isValidSCloopFactor = true;
                    }

                    if( isValidSCloopFactor == true ) {

                        correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
                        // icp.align(*unused_result, correctionCameraFrame.matrix()); 
                        // correctionCameraFrame = icp.getFinalTransformation();

                        pcl::getTranslationAndEulerAngles (correctionCameraFrame, x, y, z, roll, pitch, yaw);
                        std::cout << "x " << x << ",y " << y << ",z " << z << ",roll " << roll << " pitch " << pitch << ",yaw " << yaw << std::endl;
                        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
                        gtsam::Pose3 poseTo = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));
                        
                        // std::lock_guard<std::mutex> lock(mtx);
                        
                        float noiseScore = icp.getFitnessScore();
                        if(noiseScore < 0.06)
                        {
                            Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
                            gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
                            loopNoiseQueue.push_back(constraintNoise);
                        }               
                        else
                        {
                            noiseScore = 0.3;
                            Vector6 << noiseScore, 1e-2, noiseScore, noiseScore, noiseScore, noiseScore;
                            gtsam::noiseModel::Base::shared_ptr robust_noise = gtsam::noiseModel::Robust::Create(
                                gtsam::noiseModel::mEstimator::Cauchy::Create(1.5), // optional: replacing Cauchy by DCS or GemanMcClure
                                gtsam::noiseModel::Diagonal::Variances(Vector6)
                            );
                            loopNoiseQueue.push_back(robust_noise);
                        }
                        loopIndexQueue.push_back(make_pair(loop_idx, SCclosestHistoryFrameID_));
                        loopPoseQueue.push_back(poseFrom.between(poseTo));
                        
                        LOOPIDX loop_idx_pair{loop_idx, SCclosestHistoryFrameID_};
                        loopIndexContainer.insert(std::pair<LOOPIDX, int>(loop_idx_pair, 1));

                        if (pubIcpKeyFrames_.getNumSubscribers() != 0){
                            // pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
                            // pcl::transformPointCloud (*SClatestSurfKeyFrameCloud, *closed_cloud, correctionCameraFrame);
                            sensor_msgs::PointCloud2 cloudMsgTemp;
                            pcl::toROSMsg(*unused_result, cloudMsgTemp);
                            cloudMsgTemp.header.stamp = ros::Time().fromSec(tmp_data.pc_time);
                            cloudMsgTemp.header.frame_id = "camera_init";
                            pubIcpKeyFrames_.publish(cloudMsgTemp);
                        }
                    }
                }
            }
        }
        // data_lock_.lock();
        // *loop_path_cloud_ = *path_cloud_;
        // data_lock_.unlock();
        //pub the correct map
        PubKeyFramePose(corrected_path_cloud_, ros::Time::now().toSec());
    }


    ~LoopClosure(){}
};




int main(int argc, char** argv)
{
    ros::init(argc, argv, "loop_closure");

    LoopClosure loop_closure;

    ros::spin();

    return 0;
}








