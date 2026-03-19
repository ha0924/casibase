// Copyright 2025 The Casibase Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/casibase/casibase/i18n"
)

const minimaxAPIURL = "https://api.minimax.chat/v1/text/chatcompletion_v2"

// minimaxMessage represents a single message in the MiniMax API request/response.
type minimaxMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// minimaxRequest is the HTTP request body sent to the MiniMax API.
type minimaxRequest struct {
	Model       string           `json:"model"`
	Temperature float32          `json:"temperature"`
	Messages    []minimaxMessage `json:"messages"`
}

// minimaxChoiceMessage holds the assistant reply within a choice.
type minimaxChoiceMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// minimaxChoice represents one candidate response from the API.
type minimaxChoice struct {
	Message      minimaxChoiceMessage `json:"message"`
	FinishReason string               `json:"finish_reason"`
}

// minimaxUsage holds token consumption statistics.
type minimaxUsage struct {
	TotalTokens      int `json:"total_tokens"`
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// minimaxBaseResp carries the MiniMax API-level status code and message.
type minimaxBaseResp struct {
	StatusCode int    `json:"status_code"`
	StatusMsg  string `json:"status_msg"`
}

// minimaxResponse is the top-level structure returned by the MiniMax API.
type minimaxResponse struct {
	Choices  []minimaxChoice `json:"choices"`
	Usage    minimaxUsage    `json:"usage"`
	BaseResp minimaxBaseResp `json:"base_resp"`
}

// MiniMaxModelProvider implements the model provider interface for MiniMax.
type MiniMaxModelProvider struct {
	subType     string
	groupID     string
	apiKey      string
	temperature float32
}

// NewMiniMaxModelProvider creates a new MiniMaxModelProvider.
func NewMiniMaxModelProvider(subType string, groupID string, apiKey string, temperature float32) (*MiniMaxModelProvider, error) {
	return &MiniMaxModelProvider{
		subType:     subType,
		groupID:     groupID,
		apiKey:      apiKey,
		temperature: temperature,
	}, nil
}

// GetPricing returns the current pricing table for MiniMax models.
// Prices are per 1 000 output tokens in CNY (source: MiniMax official pricing, 2026-03).
func (p *MiniMaxModelProvider) GetPricing() string {
	return `URL:
https://api.minimax.chat/document/price

| Model                        | Price (CNY / 1k tokens) |
|------------------------------|-------------------------|
| MiniMax-M2.7                 | 0.0084                  |
| MiniMax-M2.7-highspeed       | 0.0168                  |
| MiniMax-M2.5                 | 0.0084                  |
| MiniMax-M2.5-highspeed       | 0.0168                  |
| M2-her                       | 0.0084                  |
| MiniMax-M2.1                 | 0.0084                  |
| MiniMax-M2.1-highspeed       | 0.0168                  |
| MiniMax-M2                   | 0.0084                  |
`
}

func (p *MiniMaxModelProvider) calculatePrice(modelResult *ModelResult, lang string) error {
	// priceTable maps model name to CNY per 1 000 tokens (output price).
	// Source: MiniMax official pricing page, 2026-03.
	priceTable := map[string]float64{
		"MiniMax-M2.7":           0.0084,
		"MiniMax-M2.7-highspeed": 0.0168,
		"MiniMax-M2.5":           0.0084,
		"MiniMax-M2.5-highspeed": 0.0168,
		"M2-her":                 0.0084,
		"MiniMax-M2.1":           0.0084,
		"MiniMax-M2.1-highspeed": 0.0168,
		"MiniMax-M2":             0.0084,
	}

	if pricePerThousandTokens, ok := priceTable[p.subType]; ok {
		modelResult.TotalPrice = getPrice(modelResult.TotalTokenCount, pricePerThousandTokens)
	} else {
		// 为了向后兼容，如果旧模型名（如 abab5-chat）走到这里，
		// 或者有未配置的新模型，不抛出阻断性错误，直接计价为 0。
		// 因为如果是模型名彻底废弃，上游 API 调用阶段就会拦截并返回真实报错给前端；
		// 若能走到这里说明 API 调用成功，计费模块不应产生二次误伤。
		modelResult.TotalPrice = 0
	}

	modelResult.Currency = "CNY"
	return nil
}

// QueryText sends a chat completion request to the MiniMax API using HTTP directly.
// It converts history and prompt into the standard role/content message format.
func (p *MiniMaxModelProvider) QueryText(question string, writer io.Writer, history []*RawMessage, prompt string, knowledgeMessages []*RawMessage, agentInfo *AgentInfo, lang string) (*ModelResult, error) {
	if question != "" && len(question) >= len("$CasibaseDryRun$") && question[:len("$CasibaseDryRun$")] == "$CasibaseDryRun$" {
		modelResult, err := getDefaultModelResult(p.subType, question, "")
		if err != nil {
			return nil, fmt.Errorf(i18n.Translate(lang, "model:cannot calculate tokens"))
		}
		if getContextLength(p.subType) > modelResult.TotalTokenCount {
			return modelResult, nil
		}
		return nil, fmt.Errorf(i18n.Translate(lang, "model:exceed max tokens"))
	}

	// Build message list: system prompt → history → current question.
	messages := make([]minimaxMessage, 0, len(history)+2)
	if prompt != "" {
		messages = append(messages, minimaxMessage{Role: "system", Content: prompt})
	}
	for _, msg := range history {
		role := "user"
		if msg.Author == "AI" {
			role = "assistant"
		}
		messages = append(messages, minimaxMessage{Role: role, Content: msg.Text})
	}
	messages = append(messages, minimaxMessage{Role: "user", Content: question})

	reqBody := minimaxRequest{
		Model:       p.subType,
		Temperature: p.temperature,
		Messages:    messages,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("MiniMax: failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequest(http.MethodPost, minimaxAPIURL, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("MiniMax: failed to create HTTP request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	if p.groupID != "" {
		httpReq.Header.Set("GroupId", p.groupID)
	}

	client := &http.Client{}
	httpResp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("MiniMax: HTTP request failed: %w", err)
	}
	defer httpResp.Body.Close()

	respBytes, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("MiniMax: failed to read response body: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("MiniMax: API returned HTTP %d: %s", httpResp.StatusCode, string(respBytes))
	}

	var res minimaxResponse
	if err = json.Unmarshal(respBytes, &res); err != nil {
		return nil, fmt.Errorf("MiniMax: failed to unmarshal response: %w", err)
	}

	// Guard against empty choices — include base_resp for easier debugging.
	if len(res.Choices) == 0 {
		return nil, fmt.Errorf("MiniMax: model returned empty choices, base_resp: status_code=%d status_msg=%q",
			res.BaseResp.StatusCode, res.BaseResp.StatusMsg)
	}

	flusher, ok := writer.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf(i18n.Translate(lang, "model:writer does not implement http.Flusher"))
	}

	content := res.Choices[0].Message.Content
	if _, err = fmt.Fprintf(writer, "event: message\ndata: %s\n\n", content); err != nil {
		return nil, err
	}
	flusher.Flush()

	modelResult := &ModelResult{
		TotalTokenCount:    res.Usage.TotalTokens,
		PromptTokenCount:   res.Usage.PromptTokens,
		ResponseTokenCount: res.Usage.CompletionTokens,
	}

	if err = p.calculatePrice(modelResult, lang); err != nil {
		return nil, err
	}

	return modelResult, nil
}
