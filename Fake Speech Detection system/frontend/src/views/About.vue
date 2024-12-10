<template>
  <div class="about">
    <h2>这是about页面</h2>
  </div>
</template>

<script>

export default {
  name: 'about',
  data () {
    return {
      input: ''
    }
  }
}
</script>
<style scoped>
.file-preview {
    border-radius: 5px;
    border: 1px solid #ddd;
    padding: 8px;
    width: 100%;
    margin-bottom: 5px;
    position: relative;
}
.file-drop-zone {
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.5;
    color: #212529;
    box-sizing: border-box;
    border: 1px dashed #aaa;
    border-radius: 4px;
    height: 100%;
    text-align: center;
    vertical-align: middle;
    margin: 12px 15px 12px 12px;
    padding: 5px;
}
.el-input {
    margin: 12px;
    width: 100%;
}
.input-with-select{
    background-color: #fff;
}
.el-input-group__append .el-button{
  color: #fff;
  background-color:lightgreen;
}
</style>
