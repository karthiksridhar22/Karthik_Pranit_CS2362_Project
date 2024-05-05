import socket
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import LogisticRegression
from sklearn.metrics import accuracy_score
from shutil import copyfile
import os
import shutil

def transfer_file():
    filename = "prediction"
    src_file = filename
    
    dest_dir_path = os.path.join('..', 'client_dir')
    
    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)
        print(f"Created directory: {dest_dir_path}")
    
    dest_file = os.path.join(dest_dir_path, filename)
    
    if os.path.exists(dest_file):
        print(f"{filename} already exists in {dest_dir_path}. Overwriting the file.")
        os.remove(dest_file)  
    
    try:
        shutil.copy(src_file, dest_file)
        print(f"{filename} successfully copied to {dest_dir_path}.")
    except Exception as e:
        print(f"An error occurred while copying the file: {e}")

def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"File removed: {file_path}")
    except FileNotFoundError:
        print("The file does not exist.")
    except PermissionError:
        print("Permission denied: You might not have the rights to delete the file.")
    except Exception as e:
        print(f"An error occurred: {e}")

def remove_directory_with_contents(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"All contents removed: {directory_path}")
    except FileNotFoundError:
        print("The directory does not exist.")
    except PermissionError:
        print("Permission denied: You might not have the rights to delete the directory.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_file_as_bytes(file_path):
    try:
        with open(file_path, 'rb') as file:  # Open file in binary read mode
            content = file.read()  # Read the entire content of the file as bytes
        return content
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_encrypted_data_to_array():
    # Path to the file in server_dir
    filename = 'encrypted_data'
    file_path = filename
    
    try:
        with open(file_path, 'rb') as file:
            # Read the entire file content
            file_content = file.read()
            
            # Split the content by newline delimiter
            byte_arrays = file_content.split(b"\n")
            
            # Filter out any empty byte strings resulting from consecutive newlines
            byte_arrays = [ba for ba in byte_arrays if ba]
            
            return byte_arrays
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def copy_generated_specs():
    src = os.path.join(os.getcwd(), 'generated_specs')
    dest = os.path.join(os.getcwd(), '..', 'client_dir', 'generated_specs')

    if os.path.exists(dest):
        shutil.rmtree(dest)

    try:
        shutil.copytree(src, dest)
        print("Folder successfully copied to server_dir.")
    except Exception as e:
        print(f"An error occurred: {e}")

def copy_client_zip():
    src_file = os.path.join(os.getcwd(), 'generated_specs', 'client.zip')
    dest_dir = os.path.join(os.getcwd(), '..', 'client_dir', 'client_params')
    dest_file = os.path.join(dest_dir, 'client.zip')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print("Created directory: client_params")

    if os.path.exists(dest_file):
        print("client.zip already exists in client_params. Overwriting the file.")
        os.remove(dest_file)  # Remove the existing file if you want to overwrite it
    
    try:
        shutil.copy(src_file, dest_file)
        print("client.zip successfully copied to client_params.")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_encrypted_data(data, serialized_evaluation_keys):
    encypted_prediction = FHEModelServer("generated_specs").run(data, serialized_evaluation_keys)
    return encypted_prediction



# Setup Server Socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind(('localhost', 12347))
server.listen(1)
print("Server is listening")

# Accept a client connection
client_socket, addr = server.accept()
print("Client connected")

def client_handle():

  message = "Welcome to a secure diabetes prediction service."
  client_socket.send(message.encode())

  reply = client_socket.recv(1024).decode()
  if reply.lower() == "y":
      copy_client_zip()
      message = "The key generation parameters have been sent to you. \n Please check your client directory for the client.zip file."
      client_socket.send(message.encode())

  serialized_evaluation_keys = client_socket.recv(1024)
  print(f"Received evaluation keys from client. \n {serialized_evaluation_keys}")

  client_socket.send("\n Send the encrypted data to be predicted.".encode())

  print(client_socket.recv(1024).decode())

  #load the encrypted data
  enc = read_file_as_bytes("encrypted_data")
  if (len(enc) > 0):
      print("encrypted data received")
  else:
      print("file empty")

  #compute on it
  encypted_prediction = FHEModelServer("generated_specs").run(enc, serialized_evaluation_keys)

  print("Prediction computed")

  #send the prediction back to the client
  f = open("prediction", "wb")
  f.write(encypted_prediction)
  f.close()

  transfer_file()

  client_socket.send("Prediction sent, check your directory".encode())

client_handle()


r = input(print("Would you like to delete the created files and exit? (y/n)"))
if r == "y":
    remove_file("encrypted_data")
    remove_file("prediction")
    print("Files deleted")
    client_socket.close()
