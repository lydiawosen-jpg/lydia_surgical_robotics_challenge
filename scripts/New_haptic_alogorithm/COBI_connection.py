#!/usr/bin/env python3
import socket



class COBIConnection:
    def __init__(self):
        host_ip = "10.162.34.171" # Linux desktop IP address
        port = 6400  # COBI Studio default port

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host_ip, port))
        server_socket.listen(1)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # allows port to be reused immediately after the program exits
        client_socket, client_address = server_socket.accept

        print("0 - Task started\n1 - Wire touched\n2 - Ring dropped\n3 - Task ended")
    
    def task_started(self):
        # if ring com z coordinate is greater than number, check origin of wire
        # return true or false
    

    def wire_touched(self):
        


        # return true or false 
   

    def ring_dropped(self):


        # return true or false


    def task_ended(self):
        # if ring com z axis is less than number and ring distance smaller than number, make sure wire middle protion is above strat and end points
        # return true or false


if task_started() and (self.ring_grasped_psm1 or self.ring_grasped_psm2):
    client_socker.sendall(bytes(0))
    print("Task started")
    
    if wire_touched():
        client_socker.sendall(bytes(1))
        print("Wire touched")
    
if task_started():
    if ring_dropped():
        client_socker.sendall(bytes(2))
        print("Ring dropped")
    if task_ended():
        client_socker.sendall(bytes(3))
        print("Task ended")




finally:
    client_socket.close()
    server_socket.close()