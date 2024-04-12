import { z } from "zod";
import { GENTA_API_KEY } from "$env/static/private";
import type { Endpoint } from "../endpoints";
import type { TextGenerationStreamOutput } from "@huggingface/inference";

export const endpointGentaParametersSchema = z.object({
	weight: z.number().int().positive().default(1),
	model: z.any(),
	type: z.literal("genta"),
	apiKey: z.string().default(GENTA_API_KEY)
});

export async function endpointGenta(
	input: z.input<typeof endpointGentaParametersSchema>
): Promise<Endpoint> {
	const { apiKey, model } = endpointGentaParametersSchema.parse(input);

    return async ({ messages, preprompt, generateSettings }) => {
        let messagesFormatted = messages.map((message) => ({
            role: message.from,
            content: message.content,
        }));

        if (messagesFormatted?.[0]?.role !== "system") {
            messagesFormatted = [{ role: "system", content: preprompt ?? "" }, ...messagesFormatted];
        }

        if (model.name === "Mistral-7B-Instruct-v0.2") {
            // remove system message if model is Mistral-7B-Instruct-v0.2
            messagesFormatted = messagesFormatted.filter((message) => message.role !== "system");
        }

        const parameters = { ...model.parameters, ...generateSettings };

        // print messages
        console.log("Messages:");
        messagesFormatted.forEach((message) => {
            console.log(`${message.role}: ${message.content}`);
        });

        const payload = JSON.stringify({
            model: model.id ?? model.name,
            messages: messagesFormatted,
            stream: true,
            temperature: parameters?.temperature,
            top_p: parameters?.top_p,
            max_tokens: parameters?.max_new_tokens,
        });

        const res = await fetch('https://api.genta.tech/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `${apiKey}`,
                'Content-Type': 'application/json'
            },
            body: payload
        });

        if (!res.ok) {
            throw new Error(`Failed to generate text: ${await res.text()}`);
        }
        
        // stream response is just a text stream
        const encoder = new TextDecoderStream();
        const reader = res.body?.pipeThrough(encoder).getReader();

        return (async function* () {
            let generatedText = "";
            let tokenId = 0;

            while (true) {
                const out = await reader?.read();
                if (out?.done) {
                    const output: TextGenerationStreamOutput = {
                        token: {
                            id: tokenId++,
                            text: "",
                            logprobs: 0,
                            special: true,
                        },
                        generated_text: null,
                        details: null,
                    }
                    yield output;
                    break;
                }

                const text = out?.value;
                
                const output: TextGenerationStreamOutput = {
                    token: {
                        id: tokenId++,
                        text: text,
                        logprobs: 0,
                        special: false,
                    },
                    generated_text: generatedText,
                    details: null,
                }
                yield output;
            }
        })();
    };
}

export default endpointGenta;