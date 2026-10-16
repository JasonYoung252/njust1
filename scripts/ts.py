#!/usr/bin/env python3

import rospy
from rm_msgs.msg import MoveJ, MoveJ_P, MoveL, MoveC, Plan_State
from geometry_msgs.msg import Pose

arm_dof = 6
arrival_flag = 0

def plan_state_callback(msg):
    global arrival_flag
    if msg.state:
        arrival_flag +=  1
    else:
        rospy.logerr("*******Plan State Fail arrival_flag is %d", arrival_flag)

def main():
    global arrival_flag
    
    rospy.init_node('api_move_demo')
    private_nh = rospy.get_param("~", {})
    arm_dof = private_nh.get("Arm_Dof", 6)
    rospy.loginfo("*******arm_dof = %d", arm_dof)
    
    # Publishers
    moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=10)
    moveJ_P_pub = rospy.Publisher("/rm_driver/MoveJ_P_Cmd", MoveJ_P, queue_size=10)
    moveL_pub = rospy.Publisher("/rm_driver/MoveL_Cmd", MoveL, queue_size=10)
    moveC_pub = rospy.Publisher("/rm_driver/MoveC_Cmd", MoveC, queue_size=10)
    
    # Subscriber
    rospy.Subscriber("/rm_driver/Plan_State", Plan_State, plan_state_callback)
    
    rospy.sleep(2.0)
    
    # # 1. MoveJ command
    # if arrival_flag == 0 and not rospy.is_shutdown():
    #     moveJ_msg = MoveJ()
    #     moveJ_msg.joint = [1.567970, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
    #     moveJ_msg.speed = 0.1
        
    #     rospy.loginfo("*******Plan MoveJ Start")
    #     moveJ_pub.publish(moveJ_msg)
    
    # # Wait for MoveJ completion
    # while arrival_flag == 0 and not rospy.is_shutdown():
    #     rospy.sleep(0.1)
    
    # rospy.sleep(1.0)  # Stabilization delay
    
    # 2. MoveJ_P command
    if arrival_flag == 0 and not rospy.is_shutdown():
        moveJ_P_msg = MoveJ_P()

        moveJ_P_msg.Pose.position.x = 0
        moveJ_P_msg.Pose.position.y = 0.37
        moveJ_P_msg.Pose.position.z = 0.5
        moveJ_P_msg.Pose.orientation.x = -0.653244
        moveJ_P_msg.Pose.orientation.y = -0.270923
        moveJ_P_msg.Pose.orientation.z = -0.65337
        moveJ_P_msg.Pose.orientation.w = 0.271094
        moveJ_P_msg.trajectory_connect = 0
        moveJ_P_msg.speed = 0.1

        rospy.loginfo("*******Plan MoveJ_P Start")
        moveJ_P_pub.publish(moveJ_P_msg)
    
    # Wait for MoveJ_P completion
    while arrival_flag == 0 and not rospy.is_shutdown():
        rospy.sleep(0.1)
    
    # rospy.sleep(1.0)  # Stabilization delay

    # if arrival_flag == 1 and not rospy.is_shutdown():
    #     moveJ_P_msg = MoveJ_P()
        
    #     moveJ_P_msg.Pose.position.x = 0
    #     moveJ_P_msg.Pose.position.y = 0.37
    #     moveJ_P_msg.Pose.position.z = 0.5
    #     moveJ_P_msg.Pose.orientation.x = -0.5952
    #     moveJ_P_msg.Pose.orientation.y = -0.3819
    #     moveJ_P_msg.Pose.orientation.z = -0.5948
    #     moveJ_P_msg.Pose.orientation.w = 0.3822
    #     moveJ_P_msg.trajectory_connect = 0
    #     moveJ_P_msg.speed = 0.1

    #     rospy.loginfo("*******Plan MoveJ_P Start")
    #     moveJ_P_pub.publish(moveJ_P_msg)
    
    # # Wait for MoveJ_P completion
    # while arrival_flag == 1 and not rospy.is_shutdown():
    #     rospy.sleep(0.1)
    
    # rospy.sleep(1.0)  # Stabilization delay


    # # 2. MoveJ_P command
    # if arrival_flag == 1 and not rospy.is_shutdown():
    #     moveJ_P_msg = MoveJ_P()
        
    #     moveJ_P_msg.Pose.position.x = 0.2
    #     moveJ_P_msg.Pose.position.y = 0.4
    #     moveJ_P_msg.Pose.position.z = 0.5
    #     moveJ_P_msg.Pose.orientation.x = -0.500246
    #     moveJ_P_msg.Pose.orientation.y = -0.499746
    #     moveJ_P_msg.Pose.orientation.z = -0.498855
    #     moveJ_P_msg.Pose.orientation.w = 0.501151
    #     moveJ_P_msg.trajectory_connect = 0
    #     moveJ_P_msg.speed = 0.1

    #     rospy.loginfo("*******Plan MoveJ_P Start")
    #     moveJ_P_pub.publish(moveJ_P_msg)
    
    # # Wait for MoveJ_P completion
    # while arrival_flag == 1 and not rospy.is_shutdown():
    #     rospy.sleep(0.1)
    
    # rospy.sleep(1.0)  # Stabilization delay

    # if arrival_flag == 2 and not rospy.is_shutdown():
    #     moveJ_msg = MoveJ()
    #     moveJ_msg.joint = [1.567970, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
    #     moveJ_msg.speed = 0.1
        
    #     rospy.loginfo("*******Plan MoveJ Start")
    #     moveJ_pub.publish(moveJ_msg)
    
    # # Wait for MoveJ completion
    # while arrival_flag == 2 and not rospy.is_shutdown():
    #     rospy.sleep(0.1)
    
    # rospy.sleep(1.0)  # Stabilization delay
    
    # # 2. MoveJ_P command
    # if arrival_flag == 2 and not rospy.is_shutdown():
    #     moveJ_P_msg = MoveJ_P()
    #     moveJ_P_msg.Pose.position.x = 0.2
    #     moveJ_P_msg.Pose.position.y = 0.4
    #     moveJ_P_msg.Pose.position.z = 0.5
    #     moveJ_P_msg.Pose.orientation.x = -0.50001
    #     moveJ_P_msg.Pose.orientation.y = -0.50001
    #     moveJ_P_msg.Pose.orientation.z = -0.49999
    #     moveJ_P_msg.Pose.orientation.w = 0.49999
    #     moveJ_P_msg.trajectory_connect = 0
    #     moveJ_P_msg.speed = 0.1
        
    #     rospy.loginfo("*******Plan MoveJ_P Start")
    #     moveJ_P_pub.publish(moveJ_P_msg)
    
    # # Wait for MoveJ_P completion
    # while arrival_flag == 2 and not rospy.is_shutdown():
    #     rospy.sleep(0.1)
    
    # rospy.sleep(1.0)  # Stabilization delay

    # if arrival_flag == 4 and not rospy.is_shutdown():
    #     moveJ_msg = MoveJ()
    #     moveJ_msg.joint = [1.567970, -0.801839, 2.048196, 0.010123, 0.324143, 3.122795]
    #     moveJ_msg.speed = 0.1
        
    #     rospy.loginfo("*******Plan MoveJ Start")
    #     moveJ_pub.publish(moveJ_msg)
    
    # # Wait for MoveJ completion
    # while arrival_flag == 4 and not rospy.is_shutdown():
    #     rospy.sleep(0.1)

    # # 3. MoveL command
    # if arrival_flag == 2 and not rospy.is_shutdown():
    #     moveL_msg = MoveL()
    #     moveL_msg.Pose.position.x = -0.255816
    #     moveL_msg.Pose.position.y = -0.000013
    #     moveL_msg.Pose.position.z = 0.222814
    #     moveL_msg.Pose.orientation.x = 0.995179
    #     moveL_msg.Pose.orientation.y = -0.094604
    #     moveL_msg.Pose.orientation.z = -0.025721
    #     moveL_msg.Pose.orientation.w = 0.002349
    #     moveL_msg.speed = 0.6
    #     moveL_msg.trajectory_connect = 0
        
    #     rospy.loginfo("*******Plan MoveL Start")
    #     moveL_pub.publish(moveL_msg)
    
    # # Wait for MoveL completion
    # while arrival_flag == 2 and not rospy.is_shutdown():
    #     rospy.sleep(0.1)
    
    # rospy.sleep(1.0)  # Stabilization delay
    
    # # 4. MoveC command
    # if arrival_flag == 3 and not rospy.is_shutdown():
    #     moveC_msg = MoveC()
        
    #     # Mid Pose
    #     moveC_msg.Mid_Pose.position.x = 0.400000 - 0.09110363004194966
    #     moveC_msg.Mid_Pose.position.y = -0.000009 - 0.04199112281408371
    #     moveC_msg.Mid_Pose.position.z = 0.499996
    #     moveC_msg.Mid_Pose.orientation.x = 0.708093
    #     moveC_msg.Mid_Pose.orientation.y = -0.001251
    #     moveC_msg.Mid_Pose.orientation.z = 0.706116
    #     moveC_msg.Mid_Pose.orientation.w = 0.001290
        
    #     # End Pose
    #     moveC_msg.End_Pose.position.x = 0.400000 - 0.15920310216782976
    #     moveC_msg.End_Pose.position.y = -0.000009 - 0.11564971157455597
    #     moveC_msg.End_Pose.position.z = 0.499996
    #     moveC_msg.End_Pose.orientation.x = 0.708093
    #     moveC_msg.End_Pose.orientation.y = -0.001251
    #     moveC_msg.End_Pose.orientation.z = 0.706116
    #     moveC_msg.End_Pose.orientation.w = 0.001290
        
    #     moveC_msg.speed = 0.1
    #     moveC_msg.loop = 1
    #     moveC_msg.trajectory_connect = 0
        
    #     rospy.loginfo("*******Plan MoveC Start")
    #     moveC_pub.publish(moveC_msg)
    
    # # Wait for MoveC completion
    # while arrival_flag == 3 and not rospy.is_shutdown():
    #     rospy.sleep(0.1)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass