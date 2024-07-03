<script>
    import { base } from "$app/paths";
</script>

<!-- <div class={`w-[1124px] h-[794px] flex flex-col bg-white .page-root`}> -->
<div class={`w-11/12 sm:w-[562px] xl:w-[1124px] pb-2 flex flex-col bg-white page-root overflow-hidden`}>
    <!-- Banner -->
    <div class="p-1 flex justify-center flex-col items-center mb-2">
        <a href="https://supervision.roboflow.com/latest/" target="_blank" class="flex justify-center">
            <img src="{base}/rf-supervision-cheatsheet-banner.png" alt="banner" class="w-11/12"/>
        </a>
        <div class="flex flex-row text-[0.5rem] sm:text-xs xl:text-sm justify-evenly w-full italic">
            <span>
                Author:
                <a href="https://www.linkedin.com/in/linas-kondrackis/" class="underline text-[#8315F9]" target="_blank">Linas Kondrackis</a>
                &nbsp;
                Contributors:
                <a href="https://www.linkedin.com/in/bhavay2001/" class="underline text-[#8315F9]" target="_blank">Bhavay Malhotra</a>
                &nbsp;&nbsp;
            </span>
            <span>
                supervision v0.21.0
                &nbsp;
                July 2, 2024
            </span>
        </div>

    </div>

    <!-- Columns -->
    <div class="flex-grow flex flex-col xl:flex-row justify-evenly border-">

        <!-- Col 1 -->
        <div class="shrink-0 xl:w-1/2 overflow-hidden">
            <slot name="col1"/>
        </div>

        <!-- Col 2 -->
        <div class="border-slate-200 border-l shrink-0 xl:w-1/2 overflow-hidden">
            <slot name="col2"/>
        </div>
    </div>
</div>

<style>
    .page-root {
        box-shadow: 0px 0px 16px 0px rgba(0, 0, 0, 0.1);
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        /* border-top-left-radius: 8px;
        border-top-right-radius: 8px; */
    }
</style>