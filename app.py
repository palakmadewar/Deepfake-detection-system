import os
import base64
import csv
import io
import datetime
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename

# Import custom system modules
import database
import models_handler
import pdf_generator

app = Flask(__name__)

# System Configurations
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB Max Upload Limit

# Global Adaptive Parameters
SYSTEM_SETTINGS = {
    "image_threshold": 0.50,
    "video_threshold": 0.50,
    "audio_threshold": 0.50,
    "fusion_threshold": 0.55,
    "noise_reduction": False,
    "face_crop": True,
    "video_frame_step": 10
}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'assets'), exist_ok=True)

# Initialize Database
database.init_db()

# --- WEB UI ROUTING ---

@app.route('/')
def home():
    records = database.get_records()
    total_scans = len(records)
    fake_scans = sum(1 for r in records if r['result'] == 'FAKE')
    real_scans = total_scans - fake_scans
    
    # Calculate fake ratio
    fake_ratio = (fake_scans / total_scans * 100) if total_scans > 0 else 0.0
    
    # Recent scans for summary
    recent = records[:5]
    
    return render_template('dashboard.html', 
                           total_scans=total_scans, 
                           fake_scans=fake_scans, 
                           real_scans=real_scans,
                           fake_ratio=f"{fake_ratio:.1f}%",
                           recent_scans=recent,
                           active_page='dashboard')

@app.route('/image')
def page_image():
    return render_template('image.html', settings=SYSTEM_SETTINGS, active_page='image')

@app.route('/video')
def page_video():
    return render_template('video.html', settings=SYSTEM_SETTINGS, active_page='video')

@app.route('/audio')
def page_audio():
    return render_template('audio.html', settings=SYSTEM_SETTINGS, active_page='audio')

@app.route('/url')
def page_url():
    return render_template('url.html', active_page='url')

@app.route('/qr')
def page_qr():
    return render_template('qr.html', active_page='qr')

@app.route('/history')
def page_history():
    records = database.get_records()
    return render_template('history.html', scans=records, active_page='history')

@app.route('/settings')
def page_settings():
    return render_template('settings.html', settings=SYSTEM_SETTINGS, active_page='settings')

# --- API ENDPOINTS ---

@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    return jsonify(SYSTEM_SETTINGS)

@app.route('/api/settings/update', methods=['POST'])
def api_update_settings():
    data = request.json or {}
    for key in SYSTEM_SETTINGS:
        if key in data:
            # Type casting validation
            if isinstance(SYSTEM_SETTINGS[key], bool):
                SYSTEM_SETTINGS[key] = bool(data[key])
            elif isinstance(SYSTEM_SETTINGS[key], float):
                SYSTEM_SETTINGS[key] = float(data[key])
            elif isinstance(SYSTEM_SETTINGS[key], int):
                SYSTEM_SETTINGS[key] = int(data[key])
    return jsonify({"status": "success", "settings": SYSTEM_SETTINGS})

@app.route('/api/detect/image', methods=['POST'])
def api_detect_image():
    file_path = None
    filename = "webcam_capture.jpg"
    
    # 1. Handle base64 webcam data or file upload
    if 'image_base64' in request.form:
        img_data = request.form['image_base64']
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"capture_{int(datetime.datetime.now().timestamp())}.jpg")
        with open(file_path, "wb") as fh:
            fh.write(base64.b64decode(img_data))
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{int(datetime.datetime.now().timestamp())}_{filename}")
        file.save(file_path)
    else:
        return jsonify({"error": "No image payload found"}), 400

    # 2. Run detection pipeline
    try:
        results = models_handler.predict_image(
            image_path=file_path,
            noise_reduction=SYSTEM_SETTINGS['noise_reduction'],
            face_crop=SYSTEM_SETTINGS['face_crop'],
            threshold=SYSTEM_SETTINGS['image_threshold']
        )
        
        # Save to database
        db_id = database.add_record(
            filename=filename,
            media_type='image',
            result=results['result'],
            confidence=results['confidence'],
            details=results
        )
        results['scan_id'] = db_id
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary uploaded file
        if file_path and os.path.exists(file_path) and "capture_" in file_path:
            try:
                os.remove(file_path)
            except OSError:
                pass

@app.route('/api/detect/video', methods=['POST'])
def api_detect_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file payload found"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{int(datetime.datetime.now().timestamp())}_{filename}")
    file.save(file_path)
    
    try:
        # Run video frame-by-frame analysis
        results = models_handler.predict_video(
            video_path=file_path,
            noise_reduction=SYSTEM_SETTINGS['noise_reduction'],
            face_crop=SYSTEM_SETTINGS['face_crop'],
            frame_step=SYSTEM_SETTINGS['video_frame_step'],
            threshold=SYSTEM_SETTINGS['video_threshold']
        )
        
        # Save prediction in history
        db_id = database.add_record(
            filename=filename,
            media_type='video',
            result=results['result'],
            confidence=results['confidence'],
            details=results
        )
        results['scan_id'] = db_id
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Keep video file temporarily in upload folder for session usage if needed, or remove
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

@app.route('/api/detect/audio', methods=['POST'])
def api_detect_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file payload found"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{int(datetime.datetime.now().timestamp())}_{filename}")
    file.save(file_path)
    
    try:
        # Run MFCC feature analysis
        results = models_handler.predict_audio(
            audio_path=file_path,
            threshold=SYSTEM_SETTINGS['audio_threshold']
        )
        
        # Save to database
        db_id = database.add_record(
            filename=filename,
            media_type='audio',
            result=results['result'],
            confidence=results['confidence'],
            details=results
        )
        results['scan_id'] = db_id
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

@app.route('/api/detect/fusion', methods=['POST'])
def api_detect_fusion():
    # Supports up to three uploads at once (image, video, audio)
    image_file = request.files.get('image')
    video_file = request.files.get('video')
    audio_file = request.files.get('audio')
    
    image_path = None
    video_path = None
    audio_path = None
    
    timestamp_prefix = int(datetime.datetime.now().timestamp())
    scanned_filenames = []
    
    try:
        if image_file and image_file.filename != '':
            name = secure_filename(image_file.filename)
            scanned_filenames.append(name)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"fusion_{timestamp_prefix}_{name}")
            image_file.save(image_path)
            
        if video_file and video_file.filename != '':
            name = secure_filename(video_file.filename)
            scanned_filenames.append(name)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"fusion_{timestamp_prefix}_{name}")
            video_file.save(video_path)
            
        if audio_file and audio_file.filename != '':
            name = secure_filename(audio_file.filename)
            scanned_filenames.append(name)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"fusion_{timestamp_prefix}_{name}")
            audio_file.save(audio_path)
            
        if not scanned_filenames:
            return jsonify({"error": "No media modalities uploaded for fusion."}), 400
            
        # Run Fusion decision engine
        results = models_handler.predict_fusion(
            image_path=image_path,
            video_path=video_path,
            audio_path=audio_path,
            fusion_threshold=SYSTEM_SETTINGS['fusion_threshold']
        )
        
        combined_filename = " + ".join(scanned_filenames)
        
        # Save record
        db_id = database.add_record(
            filename=combined_filename,
            media_type='fusion',
            result=results['result'],
            confidence=results['confidence'],
            details=results
        )
        results['scan_id'] = db_id
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup
        for path in [image_path, video_path, audio_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

@app.route('/api/history', methods=['GET'])
def api_history():
    media_type = request.args.get('media_type', 'all')
    records = database.get_records(media_type=media_type)
    return jsonify(records)

@app.route('/api/history/delete', methods=['POST'])
def api_history_delete():
    data = request.json or {}
    record_id = data.get('id')
    if not record_id:
        return jsonify({"error": "Missing record ID"}), 400
        
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.json or {}
    scan_id = data.get('scan_id')
    is_correct = data.get('correct')
    
    log_path = os.path.join(os.path.dirname(__file__), 'feedback_log.txt')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_path, 'a') as f:
        f.write(f"[{timestamp}] Scan ID: {scan_id} | User Feedback Correct: {is_correct}\n")
        
    print(f"[Feedback Engine] Registered feedback for scan ID {scan_id}: {is_correct}")
    return jsonify({"status": "success", "message": "Feedback successfully logged."})

@app.route('/api/history/export', methods=['GET'])
def api_history_export():
    # Exports history to a downloadable CSV
    records = database.get_records()
    
    si = io.StringIO()
    cw = csv.writer(si)
    
    # Write header
    cw.writerow(['Scan ID', 'Timestamp', 'Filename', 'Media Type', 'Verdict', 'Confidence Score', 'Model Details'])
    
    for r in records:
        details_txt = r.get('details', {}).get('mode', 'N/A')
        cw.writerow([
            r['id'],
            r['timestamp'],
            r['filename'],
            r['media_type'],
            r['result'],
            f"{r['confidence']:.4f}",
            details_txt
        ])
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=truth_shield_scan_history.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/api/report/download/<int:record_id>', methods=['GET'])
def api_report_download(record_id):
    # Fetch the history record
    record = database.get_record_by_id(record_id)
    if not record:
        return "Record not found", 404
        
    # Generate the PDF file in a temporary location
    report_filename = f"VeriFace_Report_{record_id}.pdf"
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
    
    try:
        pdf_generator.generate_pdf_report(record, report_path)
        
        # Send the file to user
        return send_file(
            report_path,
            as_attachment=True,
            download_name=report_filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        return f"Error generating PDF report: {e}", 500
    finally:
        # We can clean up the file after request has finished or keep it.
        # ReportLab creates the file. Flask's send_file keeps file handle open during transmission, 
        # so removing it immediately inside finally might error. Let's let it sit or manage it.
        pass

# Helper to support custom response mapping in flask
from flask import make_response

if __name__ == '__main__':
    # Build any missing mock files on launch
    os.system("python setup_models.py")
    app.run(debug=True, use_reloader=False, port=5000)
