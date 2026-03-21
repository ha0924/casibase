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
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/casibase/casibase/i18n"
)

type MiniMaxModelProvider struct {
	subType     string
	groupID     string
	apiKey      string
	temperature float32
	topP        float32
}

func NewMiniMaxModelProvider(subType string, groupID string, apiKey string, temperature float32, topP float32) (*MiniMaxModelProvider, error) {
	return &MiniMaxModelProvider{
		subType:     subType,
		groupID:     groupID,
		apiKey:      apiKey,
		temperature: temperature,
		topP:        topP,
	}, nil
}

func (p *MiniMaxModelProvider) GetPricing() string {
	return `URL:
https://platform.minimaxi.com/document/price

| Model                    | Input Price            | Output Price           |
|--------------------------|------------------------|------------------------|
| MiniMax-M2.7             | 2.1 CNY/1M tokens      | 8.4 CNY/1M tokens      |
| MiniMax-M2.7-highspeed   | 4.2 CNY/1M tokens      | 16.8 CNY/1M tokens     |
| MiniMax-M2.5             | 2.1 CNY/1M tokens      | 8.4 CNY/1M tokens      |
| MiniMax-M2.5-highspeed   | 4.2 CNY/1M tokens      | 16.8 CNY/1M tokens     |
| M2-her                   | 2.1 CNY/1M tokens      | 8.4 CNY/1M tokens      |
`
}

func (p *MiniMaxModelProvider) calculatePrice(modelResult *ModelResult, lang string) error {
	price := 0.0
	priceTable := map[string]float64{
		"MiniMax-M2.7":           0.0084,
		"MiniMax-M2.7-highspeed": 0.0168,
		"MiniMax-M2.5":           0.0084,
		"MiniMax-M2.5-highspeed": 0.0168,
		"M2-her":                 0.0084,
		// Legacy models retained so that old configurations do not trigger a
		// local pricing error before the upstream API has a chance to reject
		// the deprecated model name with a clear error message.
		"abab6":      0.1,
		"abab5.5":    0.015,
		"abab5-chat": 0.015,
		"abab5.5s":   0.005,
	}

	if pricePerThousandTokens, ok := priceTable[p.subType]; ok {
		price = getPrice(modelResult.TotalTokenCount, pricePerThousandTokens)
	} else {
		return fmt.Errorf(i18n.Translate(lang, "embedding:calculatePrice() error: unknown model type: %s"), p.subType)
	}

	modelResult.TotalPrice = price
	modelResult.Currency = "CNY"
	return nil
}

func (p *MiniMaxModelProvider) QueryText(question string, writer io.Writer, history []*RawMessage, prompt string, knowledgeMessages []*RawMessage, agentInfo *AgentInfo, lang string) (*ModelResult, error) {
	const BaseUrl = "https://api.minimaxi.com/v1"

	// MiniMax OpenAI-compatible API requires temperature in the range (0.0, 1.0].
	safeTemperature := p.temperature
	if safeTemperature <= 0.0 || safeTemperature > 1.0 {
		safeTemperature = 1.0
	}

	// Delegate to LocalModelProvider which handles streaming, token counting,
	// and empty-response protection via the OpenAI-compatible endpoint.
	// The groupID field is intentionally ignored: the new API authenticates
	// via Bearer token only and no longer requires GroupId.
	localProvider, err := NewLocalModelProvider("Custom", "custom-model", p.apiKey, safeTemperature, p.topP, 0, 0, BaseUrl, p.subType, 0, 0, "CNY")
	if err != nil {
		return nil, err
	}

	filterWriter := newMinimaxThinkFilter(writer)

	modelResult, err := localProvider.QueryText(question, filterWriter, history, prompt, knowledgeMessages, agentInfo, lang)
	if err != nil {
		return nil, err
	}

	err = p.calculatePrice(modelResult, lang)
	return modelResult, err
}

type minimaxThinkFilter struct {
	writer  io.Writer
	flusher http.Flusher
	inThink bool
	buf     string
}

func newMinimaxThinkFilter(writer io.Writer) *minimaxThinkFilter {
	flusher, _ := writer.(http.Flusher)
	return &minimaxThinkFilter{
		writer:  writer,
		flusher: flusher,
	}
}

func (w *minimaxThinkFilter) Write(p []byte) (n int, err error) {
	eventStr := string(p)
	const prefix = "event: message\ndata: "
	const suffix = "\n\n"

	if !strings.HasPrefix(eventStr, prefix) || !strings.HasSuffix(eventStr, suffix) {
		return w.writer.Write(p)
	}

	content := eventStr[len(prefix) : len(eventStr)-len(suffix)]
	w.buf += content

	var output strings.Builder

	for len(w.buf) > 0 {
		if !w.inThink {
			idx := strings.Index(w.buf, "<think>")
			if idx >= 0 {
				output.WriteString(w.buf[:idx])
				w.inThink = true
				w.buf = w.buf[idx+len("<think>"):]
				continue
			}

			partial := false
			for i := len("<think>") - 1; i >= 1; i-- {
				if strings.HasSuffix(w.buf, "<think>"[:i]) {
					output.WriteString(w.buf[:len(w.buf)-i])
					w.buf = w.buf[len(w.buf)-i:]
					partial = true
					break
				}
			}
			if !partial {
				output.WriteString(w.buf)
				w.buf = ""
			}
			break
		}

		idx := strings.Index(w.buf, "</think>")
		if idx >= 0 {
			w.inThink = false
			w.buf = w.buf[idx+len("</think>"):]
			for strings.HasPrefix(w.buf, "\n") {
				w.buf = w.buf[1:]
			}
			continue
		}

		partial := false
		for i := len("</think>") - 1; i >= 1; i-- {
			if strings.HasSuffix(w.buf, "</think>"[:i]) {
				w.buf = w.buf[len(w.buf)-i:]
				partial = true
				break
			}
		}
		if !partial {
			w.buf = ""
		}
		break
	}

	if output.Len() == 0 {
		return len(p), nil
	}

	_, err = fmt.Fprintf(w.writer, "%s%s%s", prefix, output.String(), suffix)
	if err != nil {
		return 0, err
	}
	return len(p), nil
}

func (w *minimaxThinkFilter) Flush() {
	if w.flusher != nil {
		w.flusher.Flush()
	}
}
