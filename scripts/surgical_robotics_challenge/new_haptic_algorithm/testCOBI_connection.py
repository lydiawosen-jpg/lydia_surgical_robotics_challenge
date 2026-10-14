#!/usr/bin/env python3
import socket
import time

# 1. Configuration (Must match your setup)
host_ip = "10.162.34.171"  # Your Linux desktop IP address
port = 6400                 # COBI Studio Port

# 2. Create and start the TCP Server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Allow the port to be reused immediately if the script restarts
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((host_ip, port))
server_socket.listen(1)

print(f"Server listening on {host_ip}:{port}...")
print("Waiting for COBI Studio on your laptop to connect...")

# 3. Accept the incoming connection from COBI
client_socket, client_address = server_socket.accept()
print(f"Connected to COBI Studio at {client_address}!")

# Helper function to send a clean integer marker
def send_marker(marker_value):
    if 0 <= marker_value <= 255:
        # Converts the number into a raw single byte and sends it
        client_socket.sendall(bytes([marker_value]))
        print(f"Sent marker value: {marker_value}")
    else:
        print("Error: Marker value must be an integer between 0 and 255.")

try:
    # Small pause to let the connection settle
    time.sleep(2)
    
    # --- SIMULATION EXPERIMENT TIMELINE EXAMPLE ---
    
    # 1. Send an experiment start marker (Value: 1)
    print("\nSimulating: Experiment starting...")
    send_marker(1) 
    time.sleep(3)   # Wait 3 seconds
    
    # 2. Send an error event marker (Value: 2)
    print("\nSimulating: User made an error in AMBF...")
    send_marker(2)
    time.sleep(3)   # Wait 3 seconds
    
    # 3. Send a success/end marker (Value: 3)
    print("\nSimulating: Target reached successfully!")
    send_marker(3)
    
    # Keep connection open briefly so COBI can process everything
    time.sleep(2)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up the sockets
    client_socket.close()
    server_socket.close()
    print("\nConnection closed safely.")