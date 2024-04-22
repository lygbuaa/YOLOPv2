import time, struct, math, enum, threading
from typing import List, Callable
import numpy as np
import rclpy
import cv2
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point32
from ros2_apa_msgs_node.msg import CompressedImageWithDet, Box2d, LanelinePts
from cv_bridge import CvBridge

g_ros2_bridge = None
g_spin_thread = None

class Ros2SensorBridge(Node):
    def __init__(self):
        super().__init__('Ros2SensorBridge')
        self._cv_bridge = CvBridge()
        self._heartbeat_pub = self.create_publisher(String, "heartbeat", 10)
        self._heartbeat_timer = self.create_timer(1.0, self.heartbeat_callback)
        self._heartbeat_count = 0
        self._ciwd_pub = self.create_publisher(CompressedImageWithDet, "/kitti/ciwd_yolopv2", 10)
        # self._img_pub = self.create_publisher(Image, "/kitti/img_laneline", 10)
        self._ciwd_counter = 0
        cv2.namedWindow("kitti", cv2.WINDOW_NORMAL)

    def __del__(self):
        cv2.destroyAllWindows()

    def heartbeat_callback(self):
        msg = String()
        msg.data = "heartbeat counter: {}".format(self._heartbeat_count)
        self._heartbeat_pub.publish(msg)
        if self._heartbeat_count % 10 == 0:
            self.get_logger().info("{:s}".format(msg.data))
        self._heartbeat_count += 1

    def add_ciwd_frame(self, timestamp:float, rgb_png_path:str, semantic_png_path:str, semantic_rgb_path:str, laneline_npy_path:str, dets_path:str, lanes_path:str) -> None:
        msg = CompressedImageWithDet()
        self._ciwd_counter += 1
        # msg.header.stamp = self.get_clock().now().to_msg()
        seconds = math.floor(timestamp)
        nanoseconds = (timestamp - seconds) * 1e9
        ts = Time(seconds=seconds, nanoseconds=nanoseconds)
        msg.header.stamp = ts.to_msg()

        img_rgb = cv2.imread(rgb_png_path)
        msg.img_height = img_rgb.shape[0]
        msg.img_width = img_rgb.shape[1]

        img_semantic = cv2.imread(semantic_png_path, cv2.IMREAD_GRAYSCALE)
        img_semantic_rgb = cv2.imread(semantic_rgb_path)
        img_semantic_half_kitti = cv2.resize(img_semantic, (640, 200))
        img_laneline = np.load(laneline_npy_path).astype(np.uint8)
        # self.get_logger().info("img_laneline: {}".format(img_laneline.shape))
        h_laneline = int(img_laneline.shape[0]*img_laneline.shape[1]/img_rgb.shape[1])
        offset_laneline = int((h_laneline-img_rgb.shape[0])*0.5)
        # img_laneline_kitti = cv2.resize(img_laneline, (img_rgb.shape[1], img_rgb.shape[0])) * 255
        img_laneline_kitti_tmp = cv2.resize(img_laneline, (img_rgb.shape[1], h_laneline)) * 255
        img_laneline_kitti = img_laneline_kitti_tmp[offset_laneline:img_rgb.shape[0]+offset_laneline, 0:img_rgb.shape[1]]
        img_laneline_half_kitti = cv2.resize(img_laneline_kitti, (640, 200))

        # self._img_pub.publish(self._cv_bridge.cv2_to_imgmsg(img_laneline_kitti, encoding="mono8"))

        msg.img_compressed = self._cv_bridge.cv2_to_compressed_imgmsg(img_rgb, dst_format="jpg")
        msg.img_semantic = self._cv_bridge.cv2_to_imgmsg(img_semantic_half_kitti, encoding="mono8")
        msg.img_laneline = self._cv_bridge.cv2_to_imgmsg(img_laneline_half_kitti, encoding="mono8")

        with open(lanes_path) as fp:
            line = fp.readline()
            lno = 0
            while line:
                vars = line.split()
                lane = LanelinePts()
                for i in range(0,len(vars),2):
                    p32 = Point32()
                    p32.x = float(vars[i])
                    p32.y = float(vars[i+1])
                    p32.z = 0.0
                    lane.points.append(p32)
                lane.id = int(lno)
                lane.num_pts = int(len(vars)/2)
                lane.type = LanelinePts.SOLID
                msg.lanelines.append(lane)
                # self.get_logger().info("lane[{}]: {}, id={}, num_pts={}, type={}".format(lno, line, lane.id, lane.num_pts, lane.type))
                line = fp.readline()
                lno += 1

        with open(dets_path) as fp:
            line = fp.readline()
            lno = 0
            while line:
                vars = line.split()
                cls = int(vars[0])
                fx = float(vars[1])
                fy = float(vars[2])
                fw = float(vars[3])
                fh = float(vars[4])
                # self.get_logger().info("bbox[{}]: {}, x={}, y={}, w={}, h={}".format(lno, line, fx, fy, fw, fh))

                ### parse xywh and fill bbox
                bbox = Box2d()
                bbox.idx = lno
                bbox.occupied = False
                bbox.viz = True
                bbox.length = fh * msg.img_height
                bbox.width = fw * msg.img_width
                bbox.center.x = fx * msg.img_width
                bbox.center.y = fy * msg.img_height
                bbox.center.theta = 0.0

                bbox.vertices[0].ix = (fx-0.5*fw) * msg.img_width
                bbox.vertices[0].iy = (fy-0.5*fh) * msg.img_height
                bbox.vertices[1].ix = (fx+0.5*fw) * msg.img_width
                bbox.vertices[1].iy = (fy-0.5*fh) * msg.img_height
                bbox.vertices[2].ix = (fx+0.5*fw) * msg.img_width
                bbox.vertices[2].iy = (fy+0.5*fh) * msg.img_height
                bbox.vertices[3].ix = (fx-0.5*fw) * msg.img_width
                bbox.vertices[3].iy = (fy+0.5*fh) * msg.img_height
                msg.objects.append(bbox)
                c1, c2 = (int(bbox.vertices[0].ix), int(bbox.vertices[0].iy)), (int(bbox.vertices[2].ix), int(bbox.vertices[2].iy))
                cv2.rectangle(img_rgb, c1, c2, [0,255,255], thickness=2, lineType=cv2.LINE_AA)

                line = fp.readline()
                lno += 1

        ### alpha blending
        alpha_fp = np.ones((msg.img_height, msg.img_width, 3), dtype=np.float32)
        alpha_rgb = alpha_fp*0.4
        alpha_semantic = alpha_fp*0.3
        alpha_laneline = alpha_fp*0.3

        img_rgb_fp = cv2.multiply(alpha_rgb, img_rgb.astype(np.float32))
        img_semantic_rgb_fp = cv2.multiply(alpha_semantic, img_semantic_rgb.astype(np.float32))
        img_laneline_fp = cv2.multiply(alpha_laneline, cv2.cvtColor(img_laneline_kitti, cv2.COLOR_GRAY2RGB).astype(np.float32))

        img_rgb_fp = cv2.add(img_rgb_fp, img_semantic_rgb_fp)
        img_rgb_fp = cv2.add(img_rgb_fp, img_laneline_fp)
        img_rgb = img_rgb_fp.astype(np.uint8)

        self.get_logger().info("ciwd_counter: {}, img_rgb: {}, img_semantic: {}-{}, img_laneline: {}".format(self._ciwd_counter, img_rgb.shape, img_semantic.shape, img_semantic[200][300], img_laneline_kitti.shape))
        self._ciwd_pub.publish(msg)
        cv2.imshow('kitti', img_rgb)
        cv2.waitKey(500)


def init_rclpy(rcl_args=None) -> None:
    if not rclpy.ok():
        rclpy.init(args=rcl_args)

def spin_rclpy_in_daemon_thread(bridge) -> None:
    bridge.get_logger().warn("rclpy spin start")
    rclpy.spin(bridge)
    bridge.get_logger().warn("rclpy spin exit")
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bridge.destroy_node()
    rclpy.shutdown()

def g_init_ros2_bridge(args=None, run_with_emulator:bool=True):
    global g_ros2_bridge
    if g_ros2_bridge:
        return g_ros2_bridge
    else:
        init_rclpy()
        g_ros2_bridge = Ros2SensorBridge()
        if not run_with_emulator:
            g_spin_thread = threading.Thread(target=spin_rclpy_in_daemon_thread, args=(g_ros2_bridge,))
            g_spin_thread.setDaemon(True)
            g_spin_thread.start()
        return g_ros2_bridge

def run_ros2_bridge_test():
    try:
        g_index = 0
        dataset_base_dir = "/home/hugoliu/github/YOLOPv2/ros2/dataset/"
        g_init_ros2_bridge()
        while True:
            kitti_index = g_index % 5
            dataset_sub_dir = dataset_base_dir + "{:d}".format(kitti_index)
            g_ros2_bridge.add_ciwd_frame(timestamp=time.time(), 
                                          rgb_png_path=dataset_sub_dir+"/rgb.png", 
                                          semantic_png_path=dataset_sub_dir+"/semantic.png", 
                                          semantic_rgb_path=dataset_sub_dir+"/semantic_rgb.png", 
                                          laneline_npy_path=dataset_sub_dir+"/laneline.npy", 
                                          dets_path=dataset_sub_dir+"/dets.txt",
                                          lanes_path=dataset_sub_dir+"/lanes.txt")
            time.sleep(1.0)
            g_index += 1
    except Exception as e:
        print("exception: {}".format(str(e)))

if __name__ == "__main__":
    print("********** running test ************")
    run_ros2_bridge_test()