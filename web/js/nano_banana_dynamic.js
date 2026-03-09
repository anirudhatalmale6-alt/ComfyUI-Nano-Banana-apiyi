import { app } from "../../scripts/app.js";

// Model capabilities: which aspect ratios and image sizes each model supports
const MODEL_CAPS = {
    // Pro models — 10 ratios, resolution depends on suffix
    "gemini-3-pro-image-preview":     { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["1K","2K","4K"] },
    "gemini-3-pro-image-preview-1k":  { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["1K"] },
    "gemini-3-pro-image-preview-2k":  { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["2K"] },
    "gemini-3-pro-image-preview-4k":  { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["4K"] },
    "gemini-3-pro-image-preview-oss": { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["1K","2K","4K"] },

    // 3.1 Flash — 14 ratios, full size range
    "gemini-3.1-flash-image-preview": { ratios: ["1:1","1:4","1:8","2:3","3:2","3:4","4:1","4:3","4:5","5:4","8:1","9:16","16:9","21:9","Auto"], sizes: ["512px","1K","2K","4K"] },

    // 2.5 Flash models — 10 ratios, max 1K
    "gemini-2.5-flash-image":             { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["1K"] },
    "gemini-2.5-flash-image-oss":         { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["1K"] },
    "gemini-2.5-flash-image-preview":     { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["1K"] },
    "gemini-2.5-flash-image-preview-oss": { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["1K"] },

    // Aliases
    "nano-banana":     { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["1K","2K","4K"] },
    "nano-banana-pro": { ratios: ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9","Auto"], sizes: ["1K","2K","4K"] },
    "nano-banana-2":   { ratios: ["1:1","1:4","1:8","2:3","3:2","3:4","4:1","4:3","4:5","5:4","8:1","9:16","16:9","21:9","Auto"], sizes: ["512px","1K","2K","4K"] },
};

const NODE_TYPES = ["NanoBananaAIO", "NanoBanana2AIO", "NanoBananaMultiTurnChat", "NanoBanana2MultiTurnChat"];

function updateComboWidget(widget, newValues, defaultValue) {
    if (!widget) return;
    widget.options.values = newValues;
    if (!newValues.includes(widget.value)) {
        widget.value = defaultValue || newValues[0];
    }
}

function applyModelCaps(node, modelName) {
    const caps = MODEL_CAPS[modelName];
    if (!caps) return;

    const ratioWidget = node.widgets.find(w => w.name === "aspect_ratio");
    const sizeWidget = node.widgets.find(w => w.name === "image_size");

    if (ratioWidget) {
        updateComboWidget(ratioWidget, caps.ratios, "1:1");
    }
    if (sizeWidget) {
        updateComboWidget(sizeWidget, caps.sizes, caps.sizes.includes("2K") ? "2K" : caps.sizes[0]);
    }

    // Trigger canvas redraw
    if (app.graph) {
        app.graph.setDirtyCanvas(true);
    }
}

app.registerExtension({
    name: "NanoBanana.DynamicWidgets",

    async nodeCreated(node) {
        if (!NODE_TYPES.includes(node.comfyClass)) return;

        const modelWidget = node.widgets.find(w => w.name === "model_name");
        if (!modelWidget) return;

        // Apply caps on initial load
        setTimeout(() => applyModelCaps(node, modelWidget.value), 100);

        // Hook into model change
        const origCallback = modelWidget.callback;
        modelWidget.callback = function(value) {
            applyModelCaps(node, value);
            if (origCallback) origCallback.call(this, value);
        };
    },
});
