import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../../components/ui/accordion";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../components/ui/select";
import { Textarea } from "../../components/ui/textarea";
import { api } from "../../controllers/API/api";
import useAlertStore from "../../stores/alertStore";
import BaseModal from "../baseModal";

export default function MagicBuildModal({
  open,
  setOpen,
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [provider, setProvider] = useState("openai");
  const [modelName, setModelName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);
  const navigate = useNavigate();

  const handleBuild = async () => {
    if (!prompt) {
      setErrorData({ title: "Error", list: ["Prompt is required"] });
      return;
    }
    setLoading(true);
    try {
      const response = await api.post("/api/v1/chat_flow/", {
        prompt,
        openai_api_key: apiKey || undefined,
        provider,
        model_name: modelName || undefined,
        base_url: baseUrl || undefined,
      });
      const flow = response.data;
      setSuccessData({ title: "Flow created successfully" });
      setOpen(false);
      navigate(`/flow/${flow.id}`);
    } catch (error) {
      console.error(error);
      const apiError = error as {
        response?: { data?: { detail?: string } };
        message?: string;
      };
      setErrorData({
        title: "Error creating flow",
        list: [
          apiError.response?.data?.detail ||
            apiError.message ||
            "Unknown error",
        ],
      });
    } finally {
      setLoading(false);
    }
  };

  const getApiKeyLabel = () => {
    switch (provider) {
      case "openai":
        return "OpenAI API Key";
      case "groq":
        return "Groq API Key";
      case "anthropic":
        return "Anthropic API Key";
      case "google":
        return "Google Gemini API Key";
      default:
        return "API Key";
    }
  };

  return (
    <BaseModal open={open} setOpen={setOpen} size="medium">
      <BaseModal.Header description="Describe the flow you want to build and let AI do the rest.">
        <span className="pr-2">Magic Build</span>
      </BaseModal.Header>
      <BaseModal.Content>
        <div className="flex flex-col gap-4 p-4">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">Prompt</label>
            <Textarea
              placeholder="e.g., Create a chatbot that translates English to French."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={3}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Provider</label>
              <Select value={provider} onValueChange={setProvider}>
                <SelectTrigger>
                  <SelectValue placeholder="Select Provider" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="groq">Groq</SelectItem>
                  <SelectItem value="anthropic">Anthropic</SelectItem>
                  <SelectItem value="google">Google Gemini</SelectItem>
                  <SelectItem value="ollama">Ollama</SelectItem>
                  <SelectItem value="mock">✨ Mock Testing (No Key)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">
                Model Name (Optional)
              </label>
              <Input
                placeholder="e.g. gpt-4o, llama3..."
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
              />
            </div>
          </div>

          {provider !== "ollama" && provider !== "mock" && (
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">
                {getApiKeyLabel()} (Optional if env var set)
              </label>
              <Input
                type="password"
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
            </div>
          )}

          <Accordion type="single" collapsible>
            <AccordionItem value="advanced">
              <AccordionTrigger>Advanced Settings</AccordionTrigger>
              <AccordionContent>
                <div className="flex flex-col gap-2 pt-2">
                  <label className="text-sm font-medium">
                    Base URL (Optional)
                  </label>
                  <Input
                    placeholder="e.g. http://localhost:11434"
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Useful for proxies or local models (like Ollama).
                  </p>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </BaseModal.Content>
      <BaseModal.Footer>
        <div className="flex w-full justify-end gap-2">
          <Button variant="outline" onClick={() => setOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleBuild} loading={loading}>
            Build Flow
          </Button>
        </div>
      </BaseModal.Footer>
    </BaseModal>
  );
}
