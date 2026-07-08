std_msgs/Header header

# High-level inspection stage reported by robot/motion control.
# Suggested convention:
# 0 idle/unknown
# 1 move_to_handle_front
# 2 press_blue_point
# 3 grasp_or_pull_handle
# 4 door_opened
# 5 recover_or_abort
int32 stage_id
string stage_name

# Current motion/control state.
string current_action
bool motion_active
float32 progress

# Robot-side health and reachability.
bool has_error
string error_code
string error_message
bool emergency_stop
bool target_reachable
string reachability_message

# Optional target identifier the robot is acting on, e.g. object_id/class_name.
string target_id
