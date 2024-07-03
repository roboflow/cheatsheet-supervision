<script lang="ts">
    export let header: string;
</script>


<!-- <div class={`w-[1123px] h-[794px] flex flex-col bg-white page-root overflow-hidden`}> -->
<div class={`w-11/12 sm:w-[562px] xl:w-[1124px] pb-2 flex flex-col bg-white page-root overflow-hidden`}>
    <!-- Header -->
    <div class="bg-[#8622FF] text-white p-2 text-xl sm:text-2xl font-inter font-bold">
        <h1>{header}</h1>
    </div>

    <!-- Columns -->
    <div class="flex-grow flex flex-col xl:flex-row justify-evenly">

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

