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

def load_data():
    # Load the dataset
    df = pd.read_csv('clinic_data.csv')
    return df

df = load_data()
df = df.to_numpy()

def append_dataframe_rows_to_list(df):
    rows_list = []
    for index in range(len(df)):
        # Append the DataFrame slice for each row to the list
        rows_list.append(df.iloc[index:index+1])
    return rows_list


def process_data(data):
  separator = ","
  string_data = data.apply(lambda row: separator.join(row.astype(str)), axis=1)
  return string_data

def make_string(data):
    string = ""
    for row in data:
        string += str(row) + "|"
    return string

def send_evaluation_keys_to_server(serialized_evaluation_keys):
    # Send the key to the server
    with open("server_dir" + "/serialized_evaluation_key.ekl", "wb") as f:
        f.write(serialized_evaluation_keys)

def transfer_file():
    filename = "encrypted_data"
    src_file = filename
    
    dest_dir_path = os.path.join('..', 'server_dir')
    
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


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 12347))

def server_handle():
  print(client.recv(1024).decode())

  inp = input(print("Would you like to request client specs? (y/n)"))
  if inp == "y":
    client.send("y".encode())
  else:
      return

  message = client.recv(1024).decode()
  print(message)

  fhemodel_client = FHEModelClient("client_params", key_dir="client_params")
  fhemodel_client.generate_private_and_evaluation_keys()
  print("Keys generated successfully")

  serialized_evaluation_keys = fhemodel_client.get_serialized_evaluation_keys()

  print("Evaluation key:", serialized_evaluation_keys)

  print("\n Sending evaluation key to server.")

  client.sendall(serialized_evaluation_keys)

  print(client.recv(1024).decode())

  f = open("encrypted_data", "ab")

  enc =  fhemodel_client.quantize_encrypt_serialize(df[[0], :])
  f.write(enc)

  f.close()

  transfer_file()
  client.send("Encrypted data sent, check your directory".encode())

  print(client.recv(1024).decode())

  enc_pred = read_file_as_bytes("prediction")
  
  print("encrypted prediction received as \n decrypting and dequantizing...")

  decrypted_prediction = fhemodel_client.deserialize_decrypt_dequantize(enc_pred)
  decrypted_prediction = np.argmax(decrypted_prediction, axis=1)
  print(f"Prediction for {df[[0],:]} is: {decrypted_prediction}")

server_handle()

r = input(print("Would you like to delete the created files and exit? (y/n)"))
if r == "y":
    remove_file("encrypted_data")
    remove_file("prediction")
    remove_directory_with_contents("client_params")
    print("Files deleted")
    
  