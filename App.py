import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

print('press q to quit')

while True:
    ret, frame = cap.read()
    if not ret:
        print('tidak ada frame')
        
    try:
        analysis = DeepFace.analyze(frame, actions = ['age','gender','race','emotion'], enforce_detection=False) 
        
        for face in analysis:
            x, y, w, h = face['region']['x'],face['region']['y'],face['region']['w'],face['region']['h']
            age = face['age']
            gender = face['dominant_gender']
            race = face['dominant_race']
            emotion = face['dominant_emotion']
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            text = f"Age: {age}, Gender: {gender}, Race: {race}, Emotion: {emotion}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    except Exception as e:
        print(e)
    
    cv2.imshow('Face Analysis', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
