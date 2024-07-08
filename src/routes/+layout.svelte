<script>
  import "../app.css";
  import { SvelteToast } from "@zerodevx/svelte-toast";
</script>

<div>
  <slot />
</div>

<SvelteToast />

<style>
  div {
    background: linear-gradient(180deg, rgba(249, 250, 251, 0) 0%, rgb(242, 246, 252) 20%),
      linear-gradient(90deg, #e2c8fe 0%, #bfdbfe 50%, #aefaff 100%);
  }
</style>
