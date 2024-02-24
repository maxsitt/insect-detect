import csv 
from datetime import datetime 
import cv2  
import requests
from image_processing import frame_norm, make_bbox_square

def store_data(frame, tracks, rec_id, rec_start, save_path, labels, save_raw_frames, save_overlay_frames, crop_bbox, four_k_resolution, webhook_url, latest_images, image_count):
   
        """Save cropped detections (+ full HQ frames) to .jpg and tracker output to metadata .csv."""
        with open(f"{save_path}/metadata_{rec_start}.csv", "a", encoding="utf-8") as metadata_file:
            metadata = csv.DictWriter(metadata_file, fieldnames=
                ["rec_ID", "timestamp", "label", "confidence", "track_ID",
                "x_min", "y_min", "x_max", "y_max", "file_path"])
            if metadata_file.tell() == 0:
                metadata.writeheader() # write header only once

            # Save full raw HQ frame (e.g. for training data collection)
            if save_raw_frames:
                for track in tracks:
                    if track == tracks[-1]:
                        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
                        raw_path = f"{save_path}/raw/{timestamp}_raw.jpg"
                        cv2.imwrite(raw_path, frame)
                        #cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            for track in tracks:
                # Don't save cropped detections if tracking status == "NEW" or "LOST" or "REMOVED"
                if track.status.name == "TRACKED":

                    # Save detections cropped from HQ frame to .jpg
                    bbox = frame_norm(frame, (track.srcImgDetection.xmin, track.srcImgDetection.ymin,
                                            track.srcImgDetection.xmax, track.srcImgDetection.ymax))
                    if crop_bbox == "square":
                        det_crop = make_bbox_square(frame, bbox, four_k_resolution)
                    else:
                        det_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    label = labels[track.srcImgDetection.label]
                    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
                    crop_path = f"{save_path}/cropped/{label}/{timestamp}_{track.id}_crop.jpg"
                    cv2.imwrite(crop_path, det_crop)
            
                    # Update the latest image for this track.id
                    latest_images[track.id] = crop_path
                    
                    # Update image count for this track.id
                    image_count[track.id] = image_count.get(track.id, 0) + 1
                    print(f"Image count for track.id {track.id}: {image_count[track.id]}")
                    
                    

                    if image_count[track.id] == 3:
                        try:
                            with open(crop_path, 'rb') as f:
                                #Open metadata CSV
                                #with open(f"{save_path}/metadata_{rec_start}.csv", 'rb') as metadata_file:
                                    # Prepare the files to be sent
                                files = {'file': f}
                                        #'metadata': ('metadata.csv', metadata_file)
                                
                                data = {
                                'accountID': 'Y7I3Jmp7dCXoank4WXKeTCSoPDp1'  # Replace with your actual account ID
                                }
                                response = requests.post(webhook_url, files=files, data=data)
                            
                                if response.status_code == 200:
                                    print(f"Successfully sent {crop_path} to webhook.")
                                else:
                                    print(f"Failed to send image to webhook. Status code: {response.status_code}")
                        except Exception as e:
                            print(f"An error occurred: {e}")

                    # Save corresponding metadata to .csv file for each cropped detection
                    data = {
                        "rec_ID": rec_id,
                        "timestamp": timestamp,
                        "label": label,
                        "confidence": round(track.srcImgDetection.confidence, 2),
                        "track_ID": track.id,
                        "x_min": round(track.srcImgDetection.xmin, 4),
                        "y_min": round(track.srcImgDetection.ymin, 4),
                        "x_max": round(track.srcImgDetection.xmax, 4),
                        "y_max": round(track.srcImgDetection.ymax, 4),
                        "file_path": crop_path
                        
                    }
                    metadata.writerow(data)
                    metadata_file.flush() # write data immediately to .csv to avoid potential data loss

                    # Save full HQ frame with overlay (bounding box, label, confidence, tracking ID) drawn on frame
                    if save_overlay_frames:
                        # Text position, font size and thickness optimized for 1920x1080 px HQ frame size
                        if not four_k_resolution:
                            cv2.putText(frame, labels[track.srcImgDetection.label], (bbox[0], bbox[3] + 28),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                            cv2.putText(frame, f"{round(track.srcImgDetection.confidence, 2)}", (bbox[0], bbox[3] + 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            cv2.putText(frame, f"ID:{track.id}", (bbox[0], bbox[3] + 92),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                        # Text position, font size and thickness optimized for 3840x2160 px HQ frame size
                        else:
                            cv2.putText(frame, labels[track.srcImgDetection.label], (bbox[0], bbox[3] + 48),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)
                            cv2.putText(frame, f"{round(track.srcImgDetection.confidence, 2)}", (bbox[0], bbox[3] + 98),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
                            cv2.putText(frame, f"ID:{track.id}", (bbox[0], bbox[3] + 164),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                        if track == tracks[-1]:
                            timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S.%f")
                            overlay_path = f"{save_path}/overlay/{timestamp}_overlay.jpg"
                            cv2.imwrite(overlay_path, frame)
                            #cv2.imwrite(overlay_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])