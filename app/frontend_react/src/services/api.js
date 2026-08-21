/**
 * API service layer.
 * All HTTP calls to the FastAPI backend go through here.
 * Components never call fetch/axios directly.
 */

import axios from "axios"

const BASE_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

const client = axios.create({
    baseURL: BASE_URL,
    timeout:60000,
    headers:{"Content-Type": "application/json"}
});

export const api = {
    health: ()=> 
        client.get("/api/v1/health"),

    startDebate: (topic , userSide) =>
        client.post("/api/v1/debate/start",{topic,user_side:userSide}),


    argue: (topic,userSide,argument,turnNumber) =>
        client.post("/api/v1/debate/argue", {
            topic,
            user_side: userSide,
            argument,
            turn_number: turnNumber,
        }),
    
    evaluateSession: ()=>
        client.get("/api/v1/debate/evaluate"),

    resetDebate: () => 
        client.post("/api/v1/debate/reset"),

    getCost: () => 
        client.get("/api/v1/debate/cost")

};
