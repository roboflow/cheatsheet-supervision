<script lang="ts">
    import Highlight from "svelte-highlight";
    import python from "svelte-highlight/languages/python";
    import bash from "svelte-highlight/languages/bash";
    import "svelte-highlight/styles/github.css";
    import { toast } from '@zerodevx/svelte-toast'
    import { base } from '$app/paths';

    export let code: string;
    export let preface: string = "";
    export let isBash: boolean = false;

    function onClickCopy() {
        navigator.clipboard.writeText(code);
        toast.push('📋 Copied to clipboard!', {
            theme: {
                '--toastBackground': '#444',
                '--toastColor': '#fff',
            }
        });
    }
</script>

<div class="bg-[#f2e5ff] mb-2">
    {#if preface}
        <p class="text-xs px-2 opacity-90">{preface}</p>
    {/if}
    <div class="bg-slate-300 relative">
        <Highlight language={isBash ? bash : python} {code} class="text-xs opacity-85"/>
        <button class="copy-button" on:click={onClickCopy}>
            <img src="{base}/copy.svg" alt="Copy" class="w-4 h-4"/>
        </button>
    </div>
</div>

<style>
    .copy-button {
        position: absolute;
        right: 0;
        top: 0;
        padding: 0.5rem 1rem;
        font-size: 0.75rem;
        border: none;
        border-radius: 0 0 0 8px;
        cursor: pointer;
        box-sizing: border-box;
        border: 1px solid transparent;
        opacity: 0.3;

        transition: background-color 0.3s;
        transition: border-color 0.3s;
        transition: opacity 0.3s;
    }

    .copy-button:hover {
        background-color: white;
        /* border: 1px solid #ca81f1; */
        opacity: 1;
    }
</style>