<template>
  <div class="navMenu">
    <template v-for="navMenu in navMenus">
      <el-menu-item v-if="!navMenu.childs&&navMenu.entity"
        :key="navMenu.entity.id"
        :data="navMenu"
        :index="navMenu.entity.name">
        <span slot="title">{{navMenu.entity.alias}}</span>
      </el-menu-item>

      <!-- 此菜单下还有子菜单 -->
      <!-- <el-submenu v-if="navMenu.childs&&navMenu.entity"
        :key="navMenu.entity.id" :data="navMenu"
        :index="navMenu.entity.name">
        <template slot="title">
          <span> {{navMenu.entity.alias}}</span>
        </template>
        <NavMenu :navMenus="navMenu.childs"></NavMenu>
      </el-submenu> -->
    </template>

  </div>
</template>

<script>
export default {
  name: 'NavMenu',
  props: ['navMenus'],
  data () {
    return {}
  },
  methods: {}
}
</script>

<style scoped>
/* 水平样式 */
.el-menu--horizontal>div {
    float: left;
    margin-left: 15%;
    margin-right: 15%;
}
 .el-menu--horizontal>div>.el-submenu {
    float: left;
}
/* 一级菜单的样式 */
.el-menu--horizontal>div>.el-menu-item {
    float: left;
    height: 60px;
    line-height: 60px;
    border-bottom: 2px solid transparent;
    color: #909399;
}
/* 解决下拉三角图标 */
.el-menu--horizontal>div>.el-submenu .el-submenu__icon-arrow{
    position: absolute;
    top: 60%;
    right: 8px;
    margin-top: -10px;
}
.el-submenu__icon-arrow {
    position: absolute;
    top: 58%;
    right: 8px;
    margin-top: -7px;
    transition: transform .3s;
    /* font-size: 12px; */
}

</style>
