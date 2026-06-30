ros2 bag record -o "Test bag" \
> /ambf/env/phantom/ring_visual/State \
> /ambf/env/phantom/wire_visual/State
echo "Recording stopped successfully."
# may collect pose for psm, video, to assess learning


 # Run your optimized minimization loop over the loaded wire segments
        for segment in self.my_total_curve:
            P0 = np.array(segment[0])
            P1 = np.array(segment[1])
            P2 = np.array(segment[2])
            P3 = np.array(segment[3])
            
            result = minimize_scalar(
                distance_objective,
                bounds=(0.0, 1.0),
                method='bounded',
                args=(ring_com, P0, P1, P2, P3)
            )
            
            if result.fun < final_result.fun:
                final_result = result
                winning_segment_points = (P0, P1, P2, P3)



closest_segment = None
        best_approx_dist = float('inf')
        
        # Fast segment pre-check to find the closest segment based on the first control point (P0)
        for segment in self.my_total_curve:
            approx_dist = np.linalg.norm(np.array(segment[0]) - ring_com)
            if approx_dist < best_approx_dist:
                best_approx_dist = approx_dist
                closest_segment = segment
       
        # Run the minimization only on the closest segment to save computation time
        P0, P1, P2, P3 = map(np.array, closest_segment)
        result = minimize_scalar(distance_objective, bounds=(0.0, 1.0), method='bounded', args=(ring_com, P0, P1, P2, P3))