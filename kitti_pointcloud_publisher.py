import os
import struct
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

def read_bin_file(bin_path):
    """读取KITTI的bin点云文件，返回N×4 numpy数组"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def pointcloud2_msg(points, frame_id='velodyne'):
    """把numpy点云转换成sensor_msgs/PointCloud2消息"""
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id

    msg.height = 1
    msg.width = points.shape[0]

    # 定义点字段，x,y,z, intensity各4字节float32
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg.is_bigendian = False
    msg.point_step = 16  # 每个点16字节(4float32)
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True

    # numpy数据转bytes，注意按照msg.point_step排列
    msg.data = points.astype(np.float32).tobytes()

    return msg

class KittiPointCloudPublisher(Node):
    def __init__(self, data_path, frame_id='velodyne', publish_frequency=10.0):
        super().__init__('kitti_pointcloud_publisher')
        self.pub = self.create_publisher(PointCloud2, '/velodyne_points', 10)
        self.data_path = data_path
        self.frame_id = frame_id
        self.files = sorted(os.listdir(data_path))
        self.index = 0
        self.timer = self.create_timer(1.0 / publish_frequency, self.timer_callback)

    def timer_callback(self):
        if self.index >= len(self.files):
            self.get_logger().info('All pointcloud frames published, shutting down.')
            rclpy.shutdown()
            return

        file_path = os.path.join(self.data_path, self.files[self.index])
        points = read_bin_file(file_path)
        msg = pointcloud2_msg(points, frame_id=self.frame_id)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.get_logger().info(f'Published frame {self.index + 1}/{len(self.files)}: {self.files[self.index]}')
        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    # 请替换成你自己的KITTI点云bin文件路径，注意是drive_0027_sync序列的velodyne_points/data文件夹
    kitti_velodyne_dir = '/home/cangzhao/ros2_ws/2011_09_30_drive_0027_sync/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data'
    node = KittiPointCloudPublisher(kitti_velodyne_dir, frame_id='velodyne', publish_frequency=10.0)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
