
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


def read_pipeline(filename):
    """
    Given input pipeline json name, parse the data section, return camera setting dict and rtsp full url.

    Arguments:
    filename -- Input pipeline json filename.
    
    Returns:
    (dict, str) -- Pair of camera setting dict and rtsp full url to access.

    """

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data_dict = json.load(file) # Deserialize the file data into a Python dictionary
            print(data_dict)
    
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
                    if cm["id"]==camera_id and "data" in cm:
                        out_dict = cm["data"]
                    if cm["id"]==net_id and "data" in cm and "settings" in cm["data"] and "location" in cm["data"]["settings"]:
                        out_url = cm["data"]["settings"]["location"]
            return out_dict, out_url

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return {}, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file. Check file format.")
        return {}, None

    return {}, None

# read_pipeline("usbcamera_pipeline.json")
