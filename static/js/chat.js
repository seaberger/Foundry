/**
 * Chamber chat — handles message submission and SSE streaming.
 * Adapted from Crucible's proven streaming patterns.
 */
(function () {
    const chatPage = document.querySelector('.chat-page');
    if (!chatPage) return;

    const sessionId = chatPage.dataset.sessionId;
    const messagesEl = document.getElementById('messages');
    const form = document.getElementById('chat-form');
    const input = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const characterName = chatPage.querySelector('h2').textContent;

    function escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/\n/g, '<br>');
    }

    function scrollToBottom() {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function addMessage(role, content) {
        const div = document.createElement('div');
        div.className = `message message-${role}`;

        const author = document.createElement('div');
        author.className = 'message-author';
        author.textContent = role === 'user' ? 'You' : characterName;

        const body = document.createElement('div');
        body.className = 'message-content';
        if (content) {
            body.innerHTML = escapeHtml(content);
        }

        div.appendChild(author);
        div.appendChild(body);
        messagesEl.appendChild(div);
        scrollToBottom();
        return body;
    }

    function setInputEnabled(enabled) {
        input.disabled = !enabled;
        sendBtn.disabled = !enabled;
        sendBtn.textContent = enabled ? 'Send' : 'Thinking\u2026';
    }

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        const message = input.value.trim();
        if (!message) return;

        // Show user message immediately
        addMessage('user', message);
        input.value = '';
        setInputEnabled(false);

        // Create assistant message placeholder with cursor
        const assistantBody = addMessage('assistant', '');
        let fullText = '';

        // Send message and stream response via SSE
        const formData = new FormData();
        formData.append('message', message);

        fetch(`/sessions/${sessionId}/message`, {
            method: 'POST',
            body: formData,
        }).then(function (response) {
            if (!response.ok) {
                throw new Error('Server returned ' + response.status);
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let done = false;

            function processChunk() {
                reader.read().then(function (result) {
                    if (result.done || done) {
                        assistantBody.innerHTML = escapeHtml(fullText);
                        setInputEnabled(true);
                        input.focus();
                        return;
                    }

                    buffer += decoder.decode(result.value, { stream: true });

                    // Parse SSE lines from buffer
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // Keep incomplete line in buffer

                    for (const line of lines) {
                        if (line.startsWith('event: done')) {
                            done = true;
                        }
                        if (line.startsWith('data: ')) {
                            const token = line.slice(6);
                            fullText += token;
                            assistantBody.innerHTML = escapeHtml(fullText) + '<span class="cursor">\u2588</span>';
                            scrollToBottom();
                        }
                    }

                    if (!done) {
                        processChunk();
                    } else {
                        assistantBody.innerHTML = escapeHtml(fullText);
                        setInputEnabled(true);
                        input.focus();
                    }
                });
            }

            processChunk();
        }).catch(function (err) {
            assistantBody.innerHTML = '<em>[Error: Could not reach the server]</em>';
            setInputEnabled(true);
        });
    });

    // Submit on Enter (Shift+Enter for newline)
    input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
    });

    // Scroll to bottom on load
    scrollToBottom();
    input.focus();
})();
