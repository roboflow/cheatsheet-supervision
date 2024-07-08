<script>
  import { base } from "$app/paths";
</script>

<div class="w-full max-w-[1123px] pb-2 flex flex-col bg-white page-root overflow-hidden rounded-lg">
  <!-- Banner -->
  <div class="h-1/2 flex justify-center flex-col items-center mb-2 py-1 px-6">
    <a href="https://supervision.roboflow.com/latest/" target="_blank" class="flex justify-center">
      <img src="{base}/rf-supervision-cheatsheet-banner.png" alt="banner" />
    </a>
    <div class="flex flex-col lg:flex-row text-sm justify-between w-full mt-2 gap-1">
      <span>
        Author:
        <a
          href="https://www.linkedin.com/in/linas-kondrackis/"
          class="underline text-[#8315F9]"
          target="_blank">Linas Kondrackis</a
        >
        &nbsp; Contributors:
        <a
          href="https://www.linkedin.com/in/bhavay2001/"
          class="underline text-[#8315F9]"
          target="_blank">Bhavay Malhotra</a
        >,
        <a
          href="https://www.linkedin.com/in/chris-doss-2a29ba1/"
          class="underline text-[#8315F9]"
          target="_blank">Chris Doss</a
        >
        &nbsp;&nbsp;
      </span>
      <span> supervision v0.21.0 &nbsp;&nbsp; July 2, 2024 </span>
    </div>
  </div>

  <!-- Columns -->
  <div class="flex-grow flex flex-col lg:flex-row lg:justify-evenly">
    <!-- Col 1 -->
    <div class="lg:w-1/2 flex flex-col gap-4">
      <slot name="col1" />
    </div>

    <!-- Col 2 -->
    <div class="border-slate-200 lg:border-l lg:w-1/2 flex flex-col gap-4">
      <slot name="col2" />
    </div>
  </div>
</div>

<style>
  .page-root {
    box-shadow: 0px 0px 16px 0px rgba(0, 0, 0, 0.1);
    /* border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px; */
    /* border-top-left-radius: 8px;
        border-top-right-radius: 8px; */
  }
</style>
