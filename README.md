# SecureEntry - Integrated Facial and Voice Recognition System

## Proposal for Enhanced Intrusion Prevention in Home Security

### Authors:

- [**Mursal Furqan Kumbhar** (Mat.: 2047419)](mailto:kumbhar.2047419@studenti.uniroma1.it "Write an email to Mursal Furqan Kumbhar")
- [**Srinjan Ghosh** (Mat.: 2053796)](mailto:ghosh.2053796@studenti.uniroma1.it "Write an Email to Srinjan Ghosh")

## Problem Statement
Traditional door authentication methods can be easily bypassed, making home security a major concern. We aim to develop an advanced intrusion alarm system that utilizes both facial and voice recognition to ensure that only authorized individuals can enter a home. This system will significantly reduce the risk of unauthorized access while providing users with a seamless experience.

## Proposed Solution
We propose a comprehensive facial and voice recognition-based intrusion alarm system that integrates with a smartphone device installed at the door. Users seeking entry will need to undergo both face and voice verification. If both verifications are successful, the door will unlock. If not, after two failed attempts, the system will activate an alarm.

## Objective
Our primary objective is to develop a dual-authentication security solution that enhances home security through advanced technology. This system will provide users with confidence in their safety while ensuring ease of access for authorized individuals.

## System Description
### Authentication Process:
1. When an individual approaches the door, they will be prompted to authenticate using the smartphone at the door.
2. The system will capture their face and voice for verification.

### Dual Verification:
- The captured face will be compared against a stored database using **OpenCV** for facial recognition.
- Simultaneously, voice recognition will authenticate the user by matching their voice to pre-recorded samples.

### Access Control:
- If both verifications match, the door will unlock.
- If either verification fails, the user will have two attempts to authenticate.
- After two unsuccessful attempts, the system will ring an alarm to alert the household.

### Smartphone Integration:
- Notifications and alerts will be sent to the homeowner’s smartphone, allowing them to monitor access attempts remotely.
- The app will provide functionalities for managing facial and voice data.

### User Management:
- Users can easily add or delete faces and voice samples through a mobile app, ensuring up-to-date security settings.

## Proposed Tech Stack
- **OpenCV**: For real-time video processing and facial recognition.
- **Python**: For developing the backend and voice recognition logic, implementing authentication.
- **React Native or Flutter**: For developing the mobile application to interface with the system.
- **SQLite or a similar database**: To securely store facial and voice data locally on the smartphone.

## Proposed Algorithms
- **Haar Cascades or DNN for Face Detection**: To identify faces in the video feed.
- **LBPH or a pre-trained deep learning model for Face Recognition**: To compare detected faces against the database.
- **Voice Recognition API**: To enable dual authentication through voice matching.
- **Alert Logic**: Conditional checks to trigger notifications or alarms based on authentication results.

## Conclusion
With the **SecureEntry** project, we aim to enhance home security through innovative dual-authentication technology. By integrating both facial and voice recognition, we can provide users with a secure and efficient entry system that protects against unauthorized access. We look forward to developing this system and exploring its potential to improve safety for all users under your kind guidance.

