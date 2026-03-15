import cv2
import numpy as np
import tensorflow as tf

# Load trained CNN model
model = tf.keras.models.load_model("digit_model.keras")

# Create black canvas
canvas = np.zeros((400, 400), dtype="uint8")
drawing = False

def draw(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 5, 255, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit", draw)


def preprocess_image(img):

    # Find digit area
    coords = cv2.findNonZero(img)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    # Resize digit
    img = cv2.resize(img, (20,20))

    # Create blank 28x28 image
    new_img = np.zeros((28,28), dtype="uint8")

    # Center digit
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    new_img[y_offset:y_offset+20, x_offset:x_offset+20] = img

    # Normalize
    new_img = new_img.astype("float32") / 255.0
    new_img = new_img.reshape(1,28,28,1)

    return new_img


while True:

    cv2.imshow("Draw Digit", canvas)
    key = cv2.waitKey(1)

    # Press p to predict
    if key == ord('p'):

        img = preprocess_image(canvas)

        prediction = model.predict(img, verbose=0)

        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        print(f"\nPredicted Digit: {digit}")
        print(f"Confidence: {confidence:.2f}%")

    # Press c to clear
    if key == ord('c'):
        canvas[:] = 0

    # Press q to quit
    if key == ord('q'):
        break


cv2.destroyAllWindows()