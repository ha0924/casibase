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

	modelResult, err := localProvider.QueryText(question, writer, history, prompt, knowledgeMessages, agentInfo, lang)
	if err != nil {
		return nil, err
	}

	err = p.calculatePrice(modelResult, lang)
	return modelResult, err
}
