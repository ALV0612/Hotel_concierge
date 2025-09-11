require('dotenv').config();
const express = require('express');
const axios = require('axios');
const crypto = require('crypto');
const app = express();

// Message deduplication cache
const processedMessages = new Set();
const MAX_CACHE_SIZE = 1000;

// Environment variables
const FB_PAGE_ACCESS_TOKEN = process.env.FB_PAGE_ACCESS_TOKEN;
const FB_VERIFY_TOKEN = process.env.FB_VERIFY_TOKEN;
const CHAT_API_URL = process.env.CHAT_API_URL;
const APP_SECRET = process.env.APP_SECRET;
const PORT = process.env.PORT || 3000;

// ========================================
// CONTEXT MANAGEMENT SYSTEM
// ========================================

class ContextManager {
  constructor() {
    this.MAX_HISTORY_MESSAGES = 3; // Giảm xuống 3 tin nhắn
    this.MAX_CONTEXT_LENGTH = 800; // Giảm xuống 800 chars để match backend limit
    this.SESSION_TIMEOUT = 30 * 60 * 1000; // 30 phút
    this.sessions = new Map(); // Session storage
  }

  // Lấy context được tối ưu hóa
  getOptimizedContext(userId, currentMessage) {
    const session = this.getSession(userId);
    
    const context = {
      session_info: this.getSessionInfo(session),
      recent_history: this.getRecentHistory(session.history),
      relevant_context: this.getRelevantContext(session.history, currentMessage)
    };

    return this.truncateContext(context);
  }

  // Lấy hoặc tạo session mới
  getSession(userId) {
    let session = this.sessions.get(userId);
    
    if (!session) {
      session = {
        history: [],
        state: {},
        lastActivity: Date.now(),
        created: Date.now()
      };
      this.sessions.set(userId, session);
    }
    
    session.lastActivity = Date.now();
    return session;
  }

  // Thêm tin nhắn vào lịch sử
  addToHistory(userId, role, content) {
    const session = this.getSession(userId);
    
    session.history.push({
      role,
      content: content.substring(0, 500), // Cắt ngắn content
      timestamp: Date.now()
    });

    // Chỉ giữ lại MAX_HISTORY_MESSAGES tin nhắn gần nhất
    if (session.history.length > this.MAX_HISTORY_MESSAGES * 2) {
      session.history = session.history.slice(-this.MAX_HISTORY_MESSAGES * 2);
    }
  }

  // Lấy thông tin session quan trọng
  getSessionInfo(session) {
    return {
      last_room_query: session.state.last_room_query || null,
      user_preferences: session.state.user_preferences || {},
      current_step: session.state.current_step || 'initial',
      session_duration: Date.now() - session.created
    };
  }

  // Chỉ lấy tin nhắn gần đây
  getRecentHistory(history) {
    if (!history || history.length === 0) return [];
    
    return history
      .slice(-this.MAX_HISTORY_MESSAGES)
      .map(msg => `${msg.role}: ${msg.content.substring(0, 150)}...`);
  }

  // Lọc context liên quan đến câu hỏi hiện tại
  getRelevantContext(history, currentQuery) {
    if (!history || !currentQuery) return null;

    const keywords = this.extractKeywords(currentQuery);
    if (!keywords.hasRelevantInfo) return null;

    const relevantMessages = history.filter(msg => 
      this.isRelevantMessage(msg.content, keywords)
    );

    return relevantMessages.length > 0 ? 
      `Relevant info: ${relevantMessages.slice(-2).map(m => m.content.substring(0, 100)).join(' | ')}` : null;
  }

  // Trích xuất keywords từ câu hỏi
  extractKeywords(query) {
    const queryLower = query.toLowerCase();
    
    // Room codes (OH101, OH102, etc.)
    const roomPattern = /oh\d+/gi;
    const rooms = query.match(roomPattern) || [];
    
    // Dates
    const datePattern = /\d{1,2}[-\/]\d{1,2}|\d{4}-\d{2}-\d{2}/g;
    const dates = query.match(datePattern) || [];
    
    // Intent detection
    let intent = 'general';
    if (queryLower.includes('chi tiết') || queryLower.includes('thông tin')) intent = 'room_details';
    else if (queryLower.includes('đặt phòng') || queryLower.includes('book')) intent = 'booking';
    else if (queryLower.includes('giá') || queryLower.includes('price')) intent = 'pricing';
    else if (queryLower.includes('phòng trống') || queryLower.includes('available')) intent = 'availability';

    return {
      rooms,
      dates,
      intent,
      hasRelevantInfo: rooms.length > 0 || dates.length > 0 || intent !== 'general'
    };
  }

  // Kiểm tra tin nhắn có liên quan không
  isRelevantMessage(content, keywords) {
    const contentLower = content.toLowerCase();
    
    // Check room codes
    if (keywords.rooms.some(room => contentLower.includes(room.toLowerCase()))) {
      return true;
    }
    
    // Check intent-specific keywords
    switch (keywords.intent) {
      case 'room_details':
        return contentLower.includes('phòng') || contentLower.includes('room') || contentLower.includes('oh');
      case 'booking':
        return contentLower.includes('đặt') || contentLower.includes('book');
      case 'pricing':
        return contentLower.includes('giá') || contentLower.includes('vnđ') || contentLower.includes('price');
      case 'availability':
        return contentLower.includes('trống') || contentLower.includes('available');
      default:
        return false;
    }
  }

  // Cắt ngắn context nếu quá dài
  truncateContext(context) {
    let contextStr = JSON.stringify(context);
    
    if (contextStr.length <= this.MAX_CONTEXT_LENGTH) return context;

    // Giảm dần recent_history
    while (context.recent_history && context.recent_history.length > 2 && 
           JSON.stringify(context).length > this.MAX_CONTEXT_LENGTH) {
      context.recent_history.shift();
    }

    // Nếu vẫn quá dài, xóa relevant_context
    if (JSON.stringify(context).length > this.MAX_CONTEXT_LENGTH) {
      context.relevant_context = null;
    }

    return context;
  }

  // Update session state từ AI response
  updateSessionState(userId, aiResponse) {
    const session = this.getSession(userId);
    
    // Parse room query info nếu có
    if (aiResponse.includes('OH') && aiResponse.includes('VNĐ')) {
      const roomInfo = this.extractRoomInfo(aiResponse);
      if (roomInfo) {
        session.state.last_room_query = roomInfo;
      }
    }
  }

  // Extract room info từ response
  extractRoomInfo(response) {
    try {
      const roomMatch = response.match(/OH\d+/g);
      const priceMatch = response.match(/[\d,]+\s*VNĐ/g);
      
      if (roomMatch && priceMatch) {
        return {
          rooms: roomMatch,
          prices: priceMatch,
          timestamp: Date.now()
        };
      }
    } catch (error) {
      console.error('Error extracting room info:', error);
    }
    return null;
  }

  // Cleanup sessions cũ
  cleanupOldSessions() {
    const now = Date.now();
    let cleanedCount = 0;

    for (const [userId, session] of this.sessions.entries()) {
      if (now - session.lastActivity > this.SESSION_TIMEOUT) {
        this.sessions.delete(userId);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      console.log(`🧹 Cleaned up ${cleanedCount} old sessions`);
    }
  }

  // Get stats
  getStats() {
    return {
      total_sessions: this.sessions.size,
      active_sessions: Array.from(this.sessions.values()).filter(
        s => Date.now() - s.lastActivity < 5 * 60 * 1000
      ).length
    };
  }
}

// Initialize Context Manager
const contextManager = new ContextManager();

// Cleanup old sessions every 10 minutes
setInterval(() => {
  contextManager.cleanupOldSessions();
}, 10 * 60 * 1000);

// ========================================
// FACEBOOK BOT HANDLERS
// ========================================

// Middleware setup
app.use(express.json({ verify: verifyRequestSignature }));

// Facebook signature verification
function verifyRequestSignature(req, res, buf) {
  const signature = req.get('X-Hub-Signature-256');
  
  if (!signature) {
    console.warn('⚠️  No signature found');
    return; // Allow requests without signature for development
  }
  
  if (!APP_SECRET) {
    console.warn('⚠️  APP_SECRET not configured, skipping signature verification');
    return;
  }
  
  try {
    const elements = signature.split('=');
    const signatureHash = elements[1];
    const expectedHash = crypto.createHmac('sha256', APP_SECRET).update(buf).digest('hex');
    
    if (signatureHash !== expectedHash) {
      throw new Error('❌ Invalid signature');
    }
  } catch (error) {
    console.error('Signature verification error:', error.message);
    throw error;
  }
}

// Webhook verification endpoint
app.get('/webhook', (req, res) => {
  const token = req.query['hub.verify_token'];
  const challenge = req.query['hub.challenge'];
  
  console.log('🔐 Webhook verification attempt:', token);
  
  if (token === FB_VERIFY_TOKEN) {
    console.log('✅ Webhook verified successfully!');
    res.send(challenge);
  } else {
    console.log('❌ Invalid verify token');
    res.sendStatus(403);
  }
});

// Webhook message handler
app.post('/webhook', async (req, res) => {
  console.log('📩 Webhook received');
  
  if (req.body.object !== 'page') {
    return res.sendStatus(404);
  }
  
  try {
    // Process all entries and messaging events
    const processingPromises = [];
    
    for (const entry of req.body.entry) {
      for (const event of entry.messaging) {
        if (event.message && event.message.text) {
          processingPromises.push(handleMessage(event));
        } else if (event.postback) {
          processingPromises.push(handlePostback(event));
        }
      }
    }
    
    await Promise.all(processingPromises);
    res.status(200).send('EVENT_RECEIVED');
    
  } catch (error) {
    console.error('Error processing webhook:', error);
    res.status(500).send('ERROR');
  }
});

// Handle incoming messages with optimized context
async function handleMessage(event) {
  const senderId = event.sender.id;
  const userMessage = event.message.text;
  const messageId = event.message.mid;
  
  // Check for duplicate messages
  if (processedMessages.has(messageId)) {
    console.log(`⚠️  Duplicate message ignored: ${messageId}`);
    return;
  }
  
  // Add to processed messages cache
  processedMessages.add(messageId);
  
  // Clean up cache if it gets too large
  if (processedMessages.size > MAX_CACHE_SIZE) {
    const oldestMessage = processedMessages.values().next().value;
    processedMessages.delete(oldestMessage);
  }
  
  console.log(`👤 User ${senderId}: ${userMessage}`);
  
  try {
    // Add user message to history
    contextManager.addToHistory(senderId, 'user', userMessage);
    
    // Show typing indicator
    await sendTypingIndicator(senderId, true);
    
    // Get optimized context
    const optimizedContext = contextManager.getOptimizedContext(senderId, userMessage);
    
    // Call chatbot API with optimized context
    console.log('🚀 Calling chatbot API with optimized context...');
    const response = await callChatbotAPI(userMessage, senderId, optimizedContext);
    
    // Hide typing indicator
    await sendTypingIndicator(senderId, false);
    
    // Extract bot response
    const botReply = extractBotResponse(response);
    console.log(`🤖 Bot reply: ${botReply.substring(0, 100)}...`);
    
    // Add bot response to history
    contextManager.addToHistory(senderId, 'assistant', botReply);
    
    // Update session state
    contextManager.updateSessionState(senderId, botReply);
    
    // Send response to Facebook
    await sendMessageToFacebook(senderId, botReply);
    
  } catch (error) {
    console.error('❌ Error handling message:', error.message);
    await sendTypingIndicator(senderId, false);
    
    const errorMessage = getErrorMessage(error);
    await sendMessageToFacebook(senderId, errorMessage);
  }
}

// Handle postback events - CHỈ GỬI PAYLOAD THUẦN
async function handlePostback(event) {
  const senderId = event.sender.id;
  const payload = event.postback.payload;
  
  console.log(`🔘 User ${senderId} clicked: ${payload}`);
  
  try {
    // Add postback to history (local tracking only)
    contextManager.addToHistory(senderId, 'user', `[POSTBACK: ${payload}]`);
    
    await sendTypingIndicator(senderId, true);
    
    // Get context for internal tracking (không gửi)
    const optimizedContext = contextManager.getOptimizedContext(senderId, payload);
    console.log(`🔍 Generated context: ${JSON.stringify(optimizedContext).length} chars (not sent)`);
    
    // Call API - CHỈ GỬI PAYLOAD
    const response = await callChatbotAPI(payload, senderId, optimizedContext);
    
    await sendTypingIndicator(senderId, false);
    
    const botReply = extractBotResponse(response);
    
    // Add bot response to history (local tracking only)
    contextManager.addToHistory(senderId, 'assistant', botReply);
    contextManager.updateSessionState(senderId, botReply);
    
    await sendMessageToFacebook(senderId, botReply);
    
  } catch (error) {
    console.error('❌ Error handling postback:', error.message);
    await sendTypingIndicator(senderId, false);
    await sendMessageToFacebook(senderId, getErrorMessage(error));
  }
}

// Call the Python chatbot API with optimized context
async function callChatbotAPI(message, userId, context) {
  const payload = {
    message: message,
    user_id: userId,
    timestamp: new Date().toISOString(),
    context: context, // Optimized context thay vì full history
    session_id: `ohana-session-${userId}`
  };

  console.log(`📦 Context size: ${JSON.stringify(context).length} characters`);
  
  const response = await axios.post(CHAT_API_URL, payload, {
    timeout: 60000,
    headers: {
      'Content-Type': 'application/json'
    }
  });
  
  return response;
}

// Extract bot response from API response
function extractBotResponse(response) {
  return response.data.bot_response || 
         response.data.response || 
         response.data.message || 
         'Xin lỗi, tôi không hiểu câu hỏi của bạn.';
}

// Get appropriate error message based on error type
function getErrorMessage(error) {
  if (error.code === 'ECONNREFUSED') {
    return 'Chatbot API không khả dụng, vui lòng thử lại sau.';
  } else if (error.code === 'ETIMEDOUT') {
    return 'Xử lý mất nhiều thời gian, vui lòng thử lại.';
  } else if (error.response && error.response.status >= 500) {
    return 'Hệ thống đang gặp sự cố, vui lòng thử lại sau ít phút.';
  } else {
    return 'Xin lỗi, tôi đang gặp sự cố kỹ thuật!';
  }
}

// Send typing indicator
async function sendTypingIndicator(recipientId, isTyping) {
  if (!FB_PAGE_ACCESS_TOKEN) {
    return;
  }
  
  const action = isTyping ? 'typing_on' : 'typing_off';
  
  try {
    await axios.post(
      `https://graph.facebook.com/v18.0/me/messages?access_token=${FB_PAGE_ACCESS_TOKEN}`,
      {
        recipient: { id: recipientId },
        sender_action: action
      },
      { timeout: 10000 }
    );
  } catch (error) {
    console.error('Failed to send typing indicator:', error.message);
  }
}

// Send message to Facebook with message splitting
async function sendMessageToFacebook(recipientId, message) {
  if (!FB_PAGE_ACCESS_TOKEN) {
    console.error('❌ FB_PAGE_ACCESS_TOKEN not configured');
    return;
  }
  
  // Split long messages
  const maxLength = 2000;
  const messages = message.length > maxLength 
    ? message.match(new RegExp(`.{1,${maxLength}}`, 'g')) 
    : [message];
  
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const messageData = {
      recipient: { id: recipientId },
      message: { text: msg.trim() }
    };
    
    try {
      await axios.post(
        `https://graph.facebook.com/v18.0/me/messages?access_token=${FB_PAGE_ACCESS_TOKEN}`,
        messageData,
        { timeout: 10000 }
      );
      console.log(`✅ Message ${i + 1}/${messages.length} sent to Facebook`);
      
      // Delay between multiple messages
      if (messages.length > 1 && i < messages.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    } catch (error) {
      console.error(`❌ Failed to send message ${i + 1}:`, 
        error.response?.data || error.message);
    }
  }
}

// Health check endpoint với context manager stats
app.get('/', (req, res) => {
  const stats = contextManager.getStats();
  
  res.json({
    status: 'running',
    timestamp: new Date().toISOString(),
    version: '2.1-optimized',
    config: {
      port: PORT,
      api_configured: !!CHAT_API_URL,
      token_configured: !!FB_PAGE_ACCESS_TOKEN,
      app_secret_configured: !!APP_SECRET,
      processed_messages_cache: processedMessages.size
    },
    context_manager: {
      total_sessions: stats.total_sessions,
      active_sessions: stats.active_sessions,
      max_history_per_session: contextManager.MAX_HISTORY_MESSAGES,
      max_context_length: contextManager.MAX_CONTEXT_LENGTH
    },
    features: {
      message_deduplication: true,
      signature_verification: !!APP_SECRET,
      postback_handling: true,
      typing_indicators: true,
      message_splitting: true,
      optimized_context_management: true,
      smart_session_cleanup: true
    }
  });
});

// Debug endpoint để xem session info
app.get('/debug/sessions', (req, res) => {
  if (process.env.NODE_ENV === 'production') {
    return res.status(404).send('Not found');
  }
  
  const sessions = {};
  for (const [userId, session] of contextManager.sessions.entries()) {
    sessions[userId] = {
      history_count: session.history.length,
      last_activity: new Date(session.lastActivity).toISOString(),
      session_age: Date.now() - session.created,
      state_keys: Object.keys(session.state)
    };
  }
  
  res.json({
    total_sessions: contextManager.sessions.size,
    sessions: sessions
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('🚨 Unhandled error:', error);
  res.status(500).json({
    error: 'Internal Server Error',
    message: error.message,
    timestamp: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  console.log('='.repeat(70));
  console.log('🌟 Facebook Chatbot Bridge v2.1-OPTIMIZED Started');
  console.log('='.repeat(70));
  console.log(`📡 Server: http://localhost:${PORT}`);
  console.log(`🔧 API URL: ${CHAT_API_URL || 'Not configured'}`);
  console.log(`🔐 FB Token: ${FB_PAGE_ACCESS_TOKEN ? 'Configured' : 'Missing'}`);
  console.log(`🛡️  App Secret: ${APP_SECRET ? 'Configured' : 'Missing'}`);
  console.log('='.repeat(70));
  console.log('✨ OPTIMIZATIONS:');
  console.log(`   🎯 Context Management: Smart filtering & truncation`);
  console.log(`   📚 History Limit: ${contextManager.MAX_HISTORY_MESSAGES} messages per session`);
  console.log(`   ⚡ Context Length: Max ${contextManager.MAX_CONTEXT_LENGTH} characters`);
  console.log(`   🧹 Auto Cleanup: ${contextManager.SESSION_TIMEOUT/60000}min timeout`);
  console.log(`   🔍 Smart Filtering: Room codes, dates, intent-based`);
  console.log('='.repeat(70));
  console.log('🚀 Ready for high-performance Facebook webhook integration!');
});