
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
// Fix: Removed LiveSession from import as it is not an exported member.
import { GoogleGenAI, LiveServerMessage, Modality, Blob, GenerateContentResponse, Part } from "@google/genai";

// --- TYPES AND INTERFACES ---
type AgentStatus = 'Active' | 'Online' | 'Busy' | 'Offline';
interface Agent {
    id: string;
    name: string;
    avatar: string;
    status: AgentStatus;
    capabilities: string[];
    config: {
        systemInstruction: string;
        model: string;
        tools?: any[];
        modelConfig?: any;
    };
}
interface Message {
    id: string;
    sender: 'You' | string; // Agent name or 'You'
    text?: string;
    image?: string; // base64 image data URL
    groundingChunks?: any[];
    isThinking?: boolean;
}
interface Chat {
    id: string;
    title: string;
    agentId: string;
    messages: Message[];
}
interface TaskStatus {
    title: string;
    step: string;
}
// Fix: Added a local interface for the LiveSession object returned by ai.live.connect().
interface LiveSession {
    close: () => void;
    sendRealtimeInput: (input: { media: Blob }) => void;
}


// --- MOCK DATA & CONFIG ---
const ALL_CAPABILITIES = {
    'Web Search': { googleSearch: {} },
    'Maps': { googleMaps: {} },
};

const INITIAL_AGENTS: Agent[] = [
    {
        id: 'codemaster',
        name: 'CodeMaster AI',
        avatar: 'https://lh3.googleusercontent.com/aida-public/AB6AXuALt_aSIB4dKeY9rHdYVus6OpWvx4HkG86Vg8qvrpJ1yaaLyrnEJjKin5yiMHWpuLhCb-gu8MIodhDWQmRRq0EqB5wKdQDvpMvLbQmfEVy4gY1fJkf3vyhIbG8ulzxGravEV6rJHm0dA6rUB1p5VSFfnYoO25-4galW0VubBrudfzMZZQYlek7Pp48UzyFLTawEVA_h9QW6qiJH6kQFBSQW1sX7_z-vYdO1jeq_U47qJM7z_7yQqKu1_TExEqmif_xJO-rKZuEQTHV5',
        status: 'Active',
        capabilities: ['Python', 'Data Analysis', 'File I/O', 'Pandas'],
        config: {
            systemInstruction: "You are CodeMaster AI, an expert software developer specializing in Python and data analysis. Provide clean, efficient, and well-documented code. When asked to perform complex tasks, think step-by-step.",
            model: 'gemini-2.5-pro',
            modelConfig: { thinkingConfig: { thinkingBudget: 32768 } }
        }
    },
    {
        id: 'researchbot',
        name: 'ResearchBot',
        avatar: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCvjJnEi50mTmPxqpYJAVgvh1zI5OprueWZdisbVZKu6ff6aD53VAet7JGlosADJJHUsPfO0w_2fh8j6m-BJn3IWEi77WH5q1UiZvAn1MYxjiL9aj8cKs1DQtndLTsOV1te8eYiQACUH2Lxdn82KVrI9TTvDZ75C2_3J6Q-02CLpUeWd8OldDY2C34PPtz_qVAJRQ5sa028xu3iMQl-rjFxBuaonJ7Cjh7xIvA0_X9YPYd96AwK8MlFs1KuuUiWCgoNY4a4YLe7o4af',
        status: 'Offline',
        capabilities: ['Web Search', 'Maps'],
        config: {
            systemInstruction: "You are ResearchBot, a helpful assistant that uses Google Search and Google Maps to provide up-to-date and accurate information. Always cite your sources.",
            model: 'gemini-2.5-flash',
            tools: [{ googleSearch: {} }, { googleMaps: {} }]
        }
    }
];

const INITIAL_CHATS: Chat[] = [
    {
        id: 'chat1',
        title: 'Python Scripting Help',
        agentId: 'codemaster',
        messages: [
            { id: 'msg1', sender: 'CodeMaster AI', text: "Sure, here is the Python script you requested for data analysis. It uses the Pandas library to read a CSV file and calculate the mean of a specific column." },
            { id: 'msg2', sender: 'You', text: "Great, can you show me how to handle potential errors if the file doesn't exist?" },
        ]
    },
    { id: 'chat2', title: 'Plan for Q3 Marketing', agentId: 'researchbot', messages: [] },
];

// --- AUDIO HELPER FUNCTIONS (for Live API) ---
const encode = (bytes: Uint8Array) => {
    let binary = '';
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

const decode = (base64: string) => {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

async function decodeAudioData(data: Uint8Array, ctx: AudioContext, sampleRate: number, numChannels: number): Promise<AudioBuffer> {
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length / numChannels;
    const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

    for (let channel = 0; channel < numChannels; channel++) {
        const channelData = buffer.getChannelData(channel);
        for (let i = 0; i < frameCount; i++) {
            channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
        }
    }
    return buffer;
}

const fileToGenerativePart = async (file: File): Promise<Part> => {
    const base64EncodedDataPromise = new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
        reader.readAsDataURL(file);
    });
    return {
        inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
    };
};


// --- REACT COMPONENT ---
const App = () => {
    // --- STATE MANAGEMENT ---
    const [agents, setAgents] = useState<Agent[]>(INITIAL_AGENTS);
    const [chats, setChats] = useState<Chat[]>(INITIAL_CHATS);
    const [selectedChatId, setSelectedChatId] = useState<string>('chat1');
    const [currentInput, setCurrentInput] = useState('');
    const [attachedFile, setAttachedFile] = useState<File | null>(null);
    const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null);
    const selectedChat = chats.find(c => c.id === selectedChatId);
    const activeAgent = agents.find(a => a.id === selectedChat?.agentId);

    // Agent Editing State
    const [isEditing, setIsEditing] = useState(false);
    const [editableAgent, setEditableAgent] = useState<Agent | undefined>(activeAgent);


    // Live API State
    const [isListening, setIsListening] = useState(false);
    const liveSessionRef = useRef<LiveSession | null>(null);
    const inputAudioContextRef = useRef<AudioContext>();
    const outputAudioContextRef = useRef<AudioContext>();
    const audioProcessorRef = useRef<ScriptProcessorNode>();
    const mediaStreamRef = useRef<MediaStream>();
    const nextStartTimeRef = useRef(0);
    const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
    
    // Sync editable agent when active agent changes
    useEffect(() => {
        if (activeAgent) {
            setEditableAgent(JSON.parse(JSON.stringify(activeAgent))); // Deep copy
            setIsEditing(false); // Reset editing state when agent changes
        }
    }, [activeAgent]);

    const updateChatMessages = (chatId: string, messages: Message[]) => {
        setChats(prevChats =>
            prevChats.map(chat =>
                chat.id === chatId ? { ...chat, messages: messages } : chat
            )
        );
    };

    // --- API CALL HANDLERS ---
    const handleSendMessage = async () => {
        if (!currentInput.trim() && !attachedFile) return;
        if (!selectedChat || !activeAgent) return;
        
        const userMessageText = currentInput;
        const userAttachedFile = attachedFile;

        setCurrentInput('');
        setAttachedFile(null);
        setTaskStatus({ title: 'Thinking...', step: 'Analyzing request' });

        const userMessage: Message = {
            id: `msg${Date.now()}`,
            sender: 'You',
            ...(userMessageText && { text: userMessageText }),
            ...(userAttachedFile && { image: URL.createObjectURL(userAttachedFile) })
        };

        const thinkingMessage: Message = {
            id: `msg${Date.now() + 1}`,
            sender: activeAgent.name,
            isThinking: true
        };

        const updatedMessages = [...selectedChat.messages, userMessage, thinkingMessage];
        updateChatMessages(selectedChat.id, updatedMessages);

        try {
            // Image Generation command
            if (userMessageText.toLowerCase().startsWith('/generate')) {
                await handleImageGeneration(userMessageText);
                return;
            }

            const promptParts: Part[] = [];
            if(userMessageText) promptParts.push({ text: userMessageText });
            if(userAttachedFile) promptParts.push(await fileToGenerativePart(userAttachedFile));

            const contents = { parts: promptParts };
            
            // Image Editing (if image is attached)
            const modelToUse = userAttachedFile && userMessageText ? 'gemini-2.5-flash-image' : activeAgent.config.model;
            
            // Fix: Restructure the generateContent call to place systemInstruction inside the config object
            // and conditionally add tools and other configs based on the model being used.
            const generateContentRequest: {
                model: string;
                contents: { parts: Part[] };
                tools?: any[];
                config?: any;
            } = {
                model: modelToUse,
                contents: contents,
            };

            if (modelToUse === 'gemini-2.5-flash-image') {
                // Image editing model does not support other configs like systemInstruction or tools.
                generateContentRequest.config = { responseModalities: [Modality.IMAGE] };
            } else {
                generateContentRequest.config = {
                    ...activeAgent.config.modelConfig,
                    systemInstruction: activeAgent.config.systemInstruction,
                };
                generateContentRequest.tools = activeAgent.config.tools;
            }

            const response = await ai.models.generateContent(generateContentRequest);

            let responseText = '';
            let responseImage = '';
            let groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
            
            if (modelToUse === 'gemini-2.5-flash-image') {
                 const part = response.candidates?.[0]?.content?.parts[0];
                 if(part?.inlineData){
                    responseImage = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                 }
            } else {
                responseText = response.text;
            }

            const aiMessage: Message = {
                id: `msg${Date.now() + 2}`,
                sender: activeAgent.name,
                ...(responseText && { text: responseText }),
                ...(responseImage && { image: responseImage }),
                ...(groundingChunks && { groundingChunks: groundingChunks })
            };
            
            updateChatMessages(selectedChat.id, [...selectedChat.messages, userMessage, aiMessage]);

        } catch (error) {
            console.error("Gemini API error:", error);
            const errorMessage: Message = { id: `err${Date.now()}`, sender: activeAgent.name, text: "Sorry, I encountered an error. Please try again." };
            updateChatMessages(selectedChat.id, [...selectedChat.messages, userMessage, errorMessage]);
        } finally {
            setTaskStatus(null);
        }
    };
    
    const handleImageGeneration = async (prompt: string) => {
        if (!selectedChat || !activeAgent) return;
        const generationPrompt = prompt.replace('/generate', '').trim();
        setTaskStatus({ title: 'Generating Image...', step: 'Sending prompt to Imagen' });

        try {
            const response = await ai.models.generateImages({
                model: 'imagen-4.0-generate-001',
                prompt: generationPrompt,
                config: { numberOfImages: 1, aspectRatio: '1:1' }
            });
            const base64ImageBytes = response.generatedImages[0].image.imageBytes;
            const imageUrl = `data:image/png;base64,${base64ImageBytes}`;
            const aiMessage: Message = { id: `img${Date.now()}`, sender: activeAgent.name, image: imageUrl };
            const finalMessages = selectedChat.messages.filter(m => !m.isThinking);
            updateChatMessages(selectedChat.id, [...finalMessages, aiMessage]);
        } catch(e) {
            console.error("Image generation failed", e);
            const errorMessage: Message = { id: `err${Date.now()}`, sender: activeAgent.name, text: "Sorry, I couldn't generate the image." };
            const finalMessages = selectedChat.messages.filter(m => !m.isThinking);
            updateChatMessages(selectedChat.id, [...finalMessages, errorMessage]);
        } finally {
            setTaskStatus(null);
        }
    };
    
    const playTTS = async (text: string) => {
        try {
            setTaskStatus({title: "Generating Speech...", step: "Please wait"});
            const response = await ai.models.generateContent({
                model: "gemini-2.5-flash-preview-tts",
                contents: [{ parts: [{ text: text }] }],
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' }}},
                },
            });

            const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            if (base64Audio) {
                const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
                const audioBuffer = await decodeAudioData(decode(base64Audio), audioContext, 24000, 1);
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.start();
            }
        } catch(e) {
            console.error("TTS failed", e);
        } finally {
            setTaskStatus(null);
        }
    };


    // --- LIVE API HANDLERS ---
    const toggleLiveConversation = useCallback(async () => {
        if (isListening) {
            // Stop listening
            liveSessionRef.current?.close();
            mediaStreamRef.current?.getTracks().forEach(track => track.stop());
            inputAudioContextRef.current?.close();
            outputAudioContextRef.current?.close();
            audioProcessorRef.current?.disconnect();
            liveSessionRef.current = null;
            setIsListening(false);
        } else {
            // Start listening
            if (!activeAgent) return;
            setIsListening(true);
            
            inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
            outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
            nextStartTimeRef.current = 0;
            audioSourcesRef.current.clear();
            const outputNode = outputAudioContextRef.current.createGain();
            outputNode.connect(outputAudioContextRef.current.destination);
            
            try {
                mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
            } catch (err) {
                console.error("Microphone access denied:", err);
                setIsListening(false);
                return;
            }

            const sessionPromise = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                callbacks: {
                    onopen: () => {
                        const source = inputAudioContextRef.current!.createMediaStreamSource(mediaStreamRef.current!);
                        const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
                        audioProcessorRef.current = scriptProcessor;
                        
                        scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                            const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                            const pcmBlob: Blob = {
                                data: encode(new Uint8Array(new Int16Array(inputData.map(v => v * 32768)).buffer)),
                                mimeType: 'audio/pcm;rate=16000',
                            };
                            sessionPromise.then((session) => {
                                session.sendRealtimeInput({ media: pcmBlob });
                            });
                        };
                        source.connect(scriptProcessor);
                        scriptProcessor.connect(inputAudioContextRef.current!.destination);
                    },
                    onmessage: async (message: LiveServerMessage) => {
                        const base64EncodedAudioString = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                        if (base64EncodedAudioString) {
                            nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputAudioContextRef.current!.currentTime);
                            const audioBuffer = await decodeAudioData(decode(base64EncodedAudioString), outputAudioContextRef.current!, 24000, 1);
                            const source = outputAudioContextRef.current!.createBufferSource();
                            source.buffer = audioBuffer;
                            source.connect(outputNode);
                            source.addEventListener('ended', () => { audioSourcesRef.current.delete(source); });
                            source.start(nextStartTimeRef.current);
                            nextStartTimeRef.current += audioBuffer.duration;
                            audioSourcesRef.current.add(source);
                        }
                        if (message.serverContent?.interrupted) {
                            for (const source of audioSourcesRef.current.values()) {
                                source.stop();
                                audioSourcesRef.current.delete(source);
                            }
                            nextStartTimeRef.current = 0;
                        }
                    },
                    onerror: (e: ErrorEvent) => {
                        console.error('Live API Error:', e);
                        setIsListening(false);
                    },
                    onclose: (e: CloseEvent) => {
                         console.log('Live API Closed');
                         setIsListening(false);
                    },
                },
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                    systemInstruction: activeAgent.config.systemInstruction,
                },
            });
            liveSessionRef.current = await sessionPromise;
        }
    }, [isListening, activeAgent]);

    // --- AGENT EDITING HANDLERS ---
    const handleSaveAgentChanges = () => {
        if (!editableAgent) return;
        setAgents(prevAgents => prevAgents.map(agent => agent.id === editableAgent.id ? editableAgent : agent));
        setIsEditing(false);
    };

    const handleCancelEdit = () => {
        if (activeAgent) {
             setEditableAgent(JSON.parse(JSON.stringify(activeAgent))); // Reset changes
        }
        setIsEditing(false);
    };

    const handleCapabilityChange = (capabilityName: string, isChecked: boolean) => {
        if (!editableAgent) return;

        const updatedTools = (editableAgent.config.tools || []).filter(tool => 
            !Object.values(ALL_CAPABILITIES).some(cap => JSON.stringify(cap) === JSON.stringify(tool))
        );

        Object.entries(ALL_CAPABILITIES).forEach(([name, tool]) => {
            const currentToolIsChecked = (name === capabilityName && isChecked) ||
                (name !== capabilityName && editableAgent.config.tools?.some(t => JSON.stringify(t) === JSON.stringify(tool)));
            
            if (currentToolIsChecked) {
                updatedTools.push(tool);
            }
        });
        
        setEditableAgent({
            ...editableAgent,
            config: {
                ...editableAgent.config,
                tools: updatedTools
            }
        });
    };
    
    // --- RENDER LOGIC ---
    if (!activeAgent || !selectedChat) {
        return <div className="flex h-screen w-full items-center justify-center">Loading...</div>;
    }

    const AgentListPanel = () => (
        <div className="flex h-screen w-16 flex-col items-center gap-4 border-r border-border-light bg-surface-subtle-light dark:border-border-dark dark:bg-surface-subtle-dark p-2">
            {agents.map(agent => (
                <button key={agent.id} title={agent.name} onClick={() => {/* TODO: Agent switching logic */}}>
                    <img src={agent.avatar} alt={agent.name} className={`h-10 w-10 rounded-full object-cover transition-all duration-200 ${activeAgent.id === agent.id ? 'ring-2 ring-primary ring-offset-2 dark:ring-offset-background-dark' : 'opacity-70 hover:opacity-100'}`} />
                </button>
            ))}
        </div>
    );

    const ChatListPanel = () => (
         <div className="hidden h-screen w-64 flex-col border-r border-border-light dark:border-border-dark md:flex">
            <div className="p-4">
                <h1 className="text-xl font-bold">{activeAgent.name}</h1>
            </div>
            <div className="flex-1 overflow-y-auto">
                <nav className="p-2">
                    {chats.filter(c => c.agentId === activeAgent.id).map(chat => (
                        <a href="#" key={chat.id} onClick={(e) => { e.preventDefault(); setSelectedChatId(chat.id); }} className={`block rounded px-3 py-2 text-sm font-medium transition-colors ${selectedChatId === chat.id ? 'bg-primary/10 text-primary' : 'hover:bg-primary/5'}`}>
                            {chat.title}
                        </a>
                    ))}
                </nav>
            </div>
        </div>
    );
    
    const ChatPanel = () => (
         <main className="flex h-screen flex-1 flex-col">
            <header className="flex h-16 items-center border-b border-border-light dark:border-border-dark px-6">
                <h2 className="text-lg font-semibold">{selectedChat.title}</h2>
            </header>
            <div className="flex-1 overflow-y-auto p-6">
                <div className="mx-auto max-w-3xl space-y-8">
                    {selectedChat.messages.map((msg) => (
                        <div key={msg.id} className={`flex items-start gap-4 ${msg.sender === 'You' ? 'justify-end' : ''}`}>
                            {msg.sender !== 'You' && <img src={activeAgent.avatar} alt={activeAgent.name} className="h-8 w-8 rounded-full" />}
                            <div className={`rounded-lg p-3 text-sm ${msg.sender === 'You' ? 'bg-primary text-white rounded-br-none' : 'bg-surface-light dark:bg-surface-dark rounded-bl-none'}`}>
                                {msg.isThinking ? (
                                    <div className="flex items-center gap-2">
                                        <div className="h-2 w-2 animate-pulse rounded-full bg-primary/50"></div>
                                        <div className="h-2 w-2 animate-pulse rounded-full bg-primary/50 [animation-delay:0.2s]"></div>
                                        <div className="h-2 w-2 animate-pulse rounded-full bg-primary/50 [animation-delay:0.4s]"></div>
                                    </div>
                                ) : (
                                    <>
                                        {msg.text && <p className="prose prose-sm dark:prose-invert max-w-none">{msg.text}</p>}
                                        {msg.image && <img src={msg.image} className="mt-2 max-w-sm rounded-md" alt="Generated content" />}
                                        {msg.groundingChunks && msg.groundingChunks.length > 0 && (
                                            <div className="mt-2 border-t border-border-light dark:border-border-dark pt-2">
                                                <h4 className="text-xs font-bold text-gray-500 dark:text-gray-400 mb-1">Sources:</h4>
                                                <ul className="list-inside list-disc text-xs">
                                                    {msg.groundingChunks.map((chunk: any, index: number) => (
                                                        <li key={index}><a href={chunk.web?.uri || chunk.maps?.uri} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">{chunk.web?.title || chunk.maps?.title || 'Source'}</a></li>
                                                    ))}
                                                </ul>
                                            </div>
                                        )}
                                        {msg.text && msg.sender !== 'You' && <button onClick={() => playTTS(msg.text!)} className="material-symbols-outlined text-sm opacity-50 hover:opacity-100 mt-2">volume_up</button>}
                                    </>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
            <div className="border-t border-border-light bg-surface-light dark:border-border-dark dark:bg-surface-dark p-4">
                <div className="relative mx-auto max-w-3xl">
                     {attachedFile && (
                        <div className="absolute bottom-full left-0 mb-2 w-full">
                            <div className="flex items-center gap-2 rounded-md border border-border-light bg-surface-subtle-light p-2 dark:border-border-dark dark:bg-surface-subtle-dark">
                                <span className="material-symbols-outlined text-lg">attachment</span>
                                <span className="text-sm truncate">{attachedFile.name}</span>
                                <button onClick={() => setAttachedFile(null)} className="material-symbols-outlined ml-auto text-sm">close</button>
                            </div>
                        </div>
                    )}
                    <form onSubmit={(e) => { e.preventDefault(); handleSendMessage(); }}>
                        <textarea value={currentInput} onChange={(e) => setCurrentInput(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }} placeholder={`Message ${activeAgent.name}...`} className="w-full resize-none rounded-lg border border-border-light bg-background-light dark:border-border-dark dark:bg-background-dark p-3 pr-28 text-sm focus:ring-primary focus:border-primary" rows={1}></textarea>
                        <div className="absolute bottom-2 right-2 flex items-center gap-1">
                            <label htmlFor="file-upload" className="cursor-pointer rounded-md p-2 hover:bg-primary/10">
                                <span className="material-symbols-outlined">attach_file</span>
                                <input id="file-upload" type="file" className="hidden" onChange={(e) => e.target.files && setAttachedFile(e.target.files[0])} />
                            </label>
                            <button type="button" onClick={toggleLiveConversation} className={`rounded-md p-2 hover:bg-primary/10 ${isListening ? 'text-red-500 animate-pulse' : 'text-primary'}`}><span className="material-symbols-outlined">{isListening ? 'mic_off' : 'mic'}</span></button>
                            <button type="submit" className="rounded-md bg-primary p-2 text-white"><span className="material-symbols-outlined">send</span></button>
                        </div>
                    </form>
                </div>
            </div>
        </main>
    );
    
    const AgentDetailsPanel = () => (
         <aside className="hidden h-screen w-80 flex-col border-l border-border-light dark:border-border-dark lg:flex">
            <div className="flex-1 overflow-y-auto p-6">
                <div className="text-center">
                    <img src={activeAgent.avatar} alt={activeAgent.name} className="mx-auto h-24 w-24 rounded-full" />
                    <h3 className="mt-4 text-xl font-bold">{activeAgent.name}</h3>
                    <p className={`mt-1 text-sm font-medium ${activeAgent.status === 'Offline' ? 'text-gray-500' : 'text-green-500'}`}>{activeAgent.status}</p>
                </div>
                
                <hr className="my-6 border-border-light dark:border-border-dark"/>

                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                         <h4 className="font-semibold">Configuration</h4>
                         {!isEditing && <button onClick={() => setIsEditing(true)} className="text-sm text-primary hover:underline">Edit</button>}
                    </div>
                    {isEditing && editableAgent ? (
                        <div className="space-y-4">
                             <div>
                                <label className="text-sm font-medium">Behavior (System Prompt)</label>
                                <textarea 
                                    value={editableAgent.config.systemInstruction}
                                    onChange={(e) => setEditableAgent({...editableAgent, config: {...editableAgent.config, systemInstruction: e.target.value }})}
                                    className="mt-1 w-full rounded-md border-border-light bg-surface-subtle-light p-2 text-sm dark:border-border-dark dark:bg-surface-subtle-dark" 
                                    rows={5}
                                />
                            </div>
                             <div>
                                <h5 className="text-sm font-medium mb-2">Capabilities</h5>
                                <div className="space-y-2">
                                    {Object.keys(ALL_CAPABILITIES).map(capName => (
                                        <label key={capName} className="flex items-center gap-2 text-sm">
                                            <input 
                                                type="checkbox" 
                                                className="rounded text-primary focus:ring-primary/50"
                                                checked={editableAgent.config.tools?.some(tool => JSON.stringify(tool) === JSON.stringify(ALL_CAPABILITIES[capName as keyof typeof ALL_CAPABILITIES]))}
                                                onChange={(e) => handleCapabilityChange(capName, e.target.checked)}
                                            />
                                            {capName}
                                        </label>
                                    ))}
                                </div>
                            </div>
                            <div className="flex items-center justify-end gap-2">
                                <button onClick={handleCancelEdit} className="rounded-md px-3 py-1 text-sm hover:bg-primary/10">Cancel</button>
                                <button onClick={handleSaveAgentChanges} className="rounded-md bg-primary px-3 py-1 text-sm text-white">Save</button>
                            </div>
                        </div>
                    ) : (
                         <>
                            <div>
                                <h5 className="text-sm font-semibold text-gray-500 dark:text-gray-400">Behavior</h5>
                                <p className="mt-1 text-sm">{activeAgent.config.systemInstruction}</p>
                            </div>
                             <div>
                                <h5 className="text-sm font-semibold text-gray-500 dark:text-gray-400 mt-4">Capabilities</h5>
                                <div className="mt-2 flex flex-wrap gap-2">
                                    {activeAgent.capabilities.map(cap => (
                                        <span key={cap} className="rounded-full bg-primary/10 px-2 py-1 text-xs font-medium text-primary">{cap}</span>
                                    ))}
                                    {activeAgent.config.tools?.map(tool => {
                                        const capName = Object.keys(ALL_CAPABILITIES).find(key => JSON.stringify(ALL_CAPABILITIES[key as keyof typeof ALL_CAPABILITIES]) === JSON.stringify(tool));
                                        return capName && <span key={capName} className="rounded-full bg-primary/10 px-2 py-1 text-xs font-medium text-primary">{capName}</span>
                                    })}
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </aside>
    );

    const TaskStatusOverlay = () => (
        taskStatus && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                <div className="rounded-lg bg-surface-light dark:bg-surface-dark p-6 text-center shadow-xl">
                    <h3 className="text-lg font-semibold">{taskStatus.title}</h3>
                    <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">{taskStatus.step}</p>
                </div>
            </div>
        )
    );

    return (
        <div className="flex h-screen w-full">
            <AgentListPanel/>
            <ChatListPanel/>
            <ChatPanel/>
            <AgentDetailsPanel/>
            <TaskStatusOverlay/>
        </div>
    );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
