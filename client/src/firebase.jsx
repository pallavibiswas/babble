// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import {getAuth} from "firebase/auth"
import { getFirestore } from 'firebase/firestore';

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyC2xuVNt8O9RDf1Q5WpyHtMD9mhtKaCCtI",
  authDomain: "babble-8e814.firebaseapp.com",
  projectId: "babble-8e814",
  storageBucket: "babble-8e814.firebasestorage.app",
  messagingSenderId: "88661710060",
  appId: "1:88661710060:web:a63d70d89612e65fe441c9",
  measurementId: "G-188ELS7E3L"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

const auth = getAuth(app)
export {auth}

export const db = getFirestore(app)

const analytics = getAnalytics(app);