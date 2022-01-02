# Pose Estimation with MoveNet Lightning on Web-Cam
## Work Flow :

- Install MoveNet Lightning -
    https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3

- Load the model
- Make detections using webcam
- Reshape input image
- Set up input and output
- Make predictions
- Draw keypoints
- Draw edges
- Real time image rendering 

### 1 ) Load the model

    interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
    interpreter.allocate_tensors() #allocating tensors to the model

### 2 ) Making detections using webcam

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        cv2.imshow('MoveNet Lightning', frame)
        
        if cv2.waitKey(10) & 0xFF==ord('q'): #q key to quit the window
            break
            
    cap.release()
    cv2.destroyAllWindows()

- These lines of code will pop-up a new window with you webcam.
- The VideoCapture device (0) can be different in your case.

### 3 ) Reshape input image

    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.float32)

- The required dimensions of input for the model is 192x192x3 represented as float32 tensor.

### 4 ) Set up input and output

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

### 5 ) Make Predictions

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

- We will store the scores for various keypoints in keypoints_with_scores variable

### Putting it all together :

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Reshaping image
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.float32)
        
        # Setting up input and output 
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Make predictions 
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        
        cv2.imshow('MoveNet Lightning', frame)
        
        if cv2.waitKey(10) & 0xFF==ord('q'): #q key to quit the window
            break
            
    cap.release()
    cv2.destroyAllWindows()

### 6 ) Draw keypoints
    def draw_keypoints(frame, keypoints, confidence_threshold):
        y, x, c = frame.shape #coordinates and confidence
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

- Here, we'll pass three key arguments : frame (image), keypoints (keypoints_with_scores), confidence_threshold (confidence)
- We will draw keypoints for the points with confidence greater than the confidence_threshold.
- We will draw the keypoints as circle with cv2.circle() function.
- In cv2.circle(), int(kx) and int(ky) represents the coordinates of the keypoint in integer, 4 represents how big we want our keypoints to be, (0, 255, 0) represents the color (green in this case) and -1 represents the thickness of the keypoint.

### 7 ) Draw edges
    def draw_connections(frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape #coordinates and confidence
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        
        for edge, color in edges.items():
            p1, p2 = edge 
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            if(c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

- Here, we'll pass four key arguments : frame (image), keypoints (keypoints_with_scores),edges (with color), confidence_threshold (confidence)
- We will draw edges for the points with confidence greater than the confidence_threshold.
- We will draw the edges as line with cv2.line() function.
- In cv2.line(), int(x1), int(y1) and int(x2), int(y2) represents the coordinates of two keypoints in integer, (0, 0, 255) represents the color (red in this case) and 2 represents the thickness of the line.

### Edges with color
    EDGES = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }


### 8 )Real time image rendering
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Reshaping image
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
        input_image = tf.cast(img, dtype=tf.float32)
        
        # Setting up input and output 
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Make predictions 
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index']) #scores for various keypoints
        
        # Rendering 
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4) #taking confidence threshold of 0.4 or 40%
        draw_keypoints(frame, keypoints_with_scores, 0.4) #taking confidence threshold of 0.4 or 40%
        
        cv2.imshow('MoveNet Lightning', frame)
        
        if cv2.waitKey(10) & 0xFF==ord('q'): #q key to quit the window
            break
            
    cap.release()
    cv2.destroyAllWindows()

