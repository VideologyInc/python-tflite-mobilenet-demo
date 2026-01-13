#!/usr/bin/env python

"""

Pipeline json file I/O.

Copyright (C) 2026 Videology
Programmed by Jianping Ye <jye@videologyinc.com>
  
Jan 026. Added save_pipeline() for any changes by user input, especially port changes.

"""

import json

from urllib.parse import urlparse

def get_port_from_url(url_string):
    """
    Extracts the port number from a given URL string.
    Returns the port number (integer) or None if no port is specified 
    and it is not a well-known scheme (like http/https).
    """
    parsed_url = urlparse(url_string)
    
    if parsed_url.port:
        return parsed_url.port
    elif parsed_url.scheme in ('http', 'ws'):
        return 80
    elif parsed_url.scheme in ('https', 'wss'):
        return 443
    else:
        # Returns None if no port is explicitly mentioned and the scheme is non-standard
        return None

def get_processing_nodes(data_dict):
    """
    Given full data dict from pipeline json file, extract its processing nodes as output

    Arguments:
    data_dict (dict) -- Input pipeline full data dict.
    
    Returns:
    (list[dict]) -- List of processing node dict ('data' section in the full data dict).

    """

    process_list = []
    if "processingId" in data_dict:
        node_list = data_dict["processingId"]
        # Go through components to find matching node of each node id.
        for id in node_list:
            if "components" in data_dict:
                for cm in data_dict["components"]:
                    if "id" in cm and cm["id"]==id and "data" in cm:
                        process_list.append(cm["data"])
    return process_list


def read_pipeline(filename):
    """
    Given input pipeline json name, parse the data section, return truple of (full data dict, camera setting dict, rtsp full url).

    Arguments:
    filename (str) -- Input pipeline json filename.
    
    Returns:
    (dict, dict, str) -- Tuple of full data dict, camera setting dict and rtsp full url to access.

    """

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data_dict = json.load(file) # Deserialize the file data into a Python dictionary
            # print(data_dict)
    
        # standard Scailx Portal pipeline file should contain "inputId" and "components" / "data" section.
        if "inputId" in data_dict:
            camera_id = data_dict["inputId"]
        else:
            camera_id = "local_camera0"
        
        net_id = "input_network_stream0"    # should also add to pipeline file bottom ;-) 

        # Try to find matching "node" in components and extract its "data" section as output dict ;-)
        if "components" in data_dict:
            out_dict = {}
            out_url = ""
            for cm in data_dict["components"]:
                if "id" in cm:
                    if cm["id"]==camera_id and "data" in cm and "settings" in cm["data"]:
                        out_dict = cm["data"]["settings"]
                    if cm["id"]==net_id and "data" in cm and "settings" in cm["data"] and "location" in cm["data"]["settings"]:
                        out_url = cm["data"]["settings"]["location"]
            print("Load camera settings from pipeline json file ", filename)
            return data_dict, out_dict, out_url

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return {}, {}, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file. Check file format.")
        return {}, {}, None

    return {}, {}, None


def save_pipeline(filename, data_dict, pipe_dict, net_url):
    """
    Given output pipeline json name, original full data dict, camera settings pipe_dict, and old net_url, update changes to full dict and save.

    Arguments:
    filename (str) -- Output pipeline json filename.
    data_dict (dict) -- Full data dict from original pipeline json file.
    pipe_dict (dict) -- Camera settings dict extracted from data_dict, plus 'port' string.
    net_url (str) -- Old net url extracted from data_dict. 

    Returns:
    (bool) -- True or False.

    """

    # Replace device_url port
    port = get_port_from_url(net_url)
    print("old url port = ", port)
    print("user input port = ", pipe_dict["port"])    
    new_url = net_url.replace(":" + str(port), ":" + pipe_dict["port"])
    print("new url = ", new_url)

    if "inputId" in data_dict:
        camera_id = data_dict["inputId"]
    else:
        camera_id = "local_camera0"
    net_id = "input_network_stream0"    # should also add to pipeline file bottom ;-) 

    if "components" in data_dict:
        # data_dict["components"] is a list
        n = len(data_dict["components"])
        for i in range(0,n):
            cm = data_dict["components"][i]
            if "id" in cm:
                if cm["id"]==camera_id and "data" in cm and "settings" in cm["data"]:
                    data_dict["components"][i]["data"]["settings"] = pipe_dict
                if cm["id"]==net_id and "data" in cm and "settings" in cm["data"] and "location" in cm["data"]["settings"]:
                    data_dict["components"][i]["data"]["settings"]["location"] = new_url

    # print()
    # print(data_dict)
    # print()

    # Save data_dict to output json file.
    with open(filename, "w") as write_file:
        # Use json.dump with indent=4 for pretty printing
        json.dump(data_dict, write_file, indent=4)
        return True
    
    return False