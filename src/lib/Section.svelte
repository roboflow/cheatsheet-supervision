<script lang="ts">
  export let header: string;
</script>

<div class="flex flex-col gap-4">
  <h2 class="text-xl font-inter">
    <span class="text-[#8622FF] ml-1 mb-1">›</span>
    {header}
  </h2>
  <slot />
</div>
