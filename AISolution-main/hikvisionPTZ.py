import requests
from requests.auth import HTTPDigestAuth
import re

def go_to_position(url, channel, id, password, pan, tilt, zoom):
    GoToURL = f'http://{url}/ISAPI/PTZCtrl/channels/{channel}/absolute'
    xml_payload = f'''<PTZData>
                        <AbsoluteHigh>
                            <elevation>{tilt}</elevation>
                            <azimuth>{pan}</azimuth>
                            <absoluteZoom>{zoom}</absoluteZoom>
                        </AbsoluteHigh>
                      </PTZData>'''
    # <AbsoluteHigh><!--high-accuracy positioning which is accurate to one decimal place-->

    response = requests.put(GoToURL, auth=HTTPDigestAuth(id, password), data=xml_payload)
    if response.status_code == 200:
        print('Moved to PTZ successfully')
    else:
        print(f'Failed to move to PTZ: {response.status_code}')

def get_position(url,channel,id, password):
    StatusURL = f'http://{url}/ISAPI/PTZCtrl/channels/{channel}/status'
    response = requests.get(StatusURL, auth=HTTPDigestAuth(id, password))
    
    if response.status_code == 200:
        print('Status recived successfully')
        patern = r"<[absoluteZoom,azimuth,elevation]+>\d+</[absoluteZoom,azimuth,elevation]+>"
        ptzData = re.findall(patern, response.text)
        # print(ptzData)

        tilt_position = int(re.findall(r"\d+", ptzData[0])[0]) # elevation
        pan_position  = int(re.findall(r"\d+", ptzData[1])[0]) # azimuth        
        zoom_position = int(re.findall(r"\d+", ptzData[2])[0]) # zoom

    else:
        print(f'Failed to move to PTZ: {response.status_code}')

    return pan_position, tilt_position, zoom_position
   