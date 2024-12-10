<template>
  <div id="app">
    <el-container class="home-container">
      <!-- 头部区域 -->
      <el-header>
        <el-menu
          :default-active="activeIndex"
          class="el-menu"
          mode="horizontal"
          @select="handleSelect"
          background-color="#343A40"
          text-color="#CCCDCF"
          active-text-color="#FFFFFF"
          :router="true"
        >
        <NavMenu :navMenus="menuData"></NavMenu>
        </el-menu>
      </el-header>
      <!-- 页面主体区域 -->
      <el-main>
        <router-view></router-view>
      </el-main>
    </el-container>
  </div>
</template>

<script>

import NavMenu from './components/NavMenu.vue'

export default {
  name: 'app',
  components: {
    NavMenu: NavMenu
  },
  data () {
    return {
      activeIndex: 'home',
      menuData: [
        {
          entity: {
            id: 1,
            name: 'home',
            alias: 'Home'
          }
        },
        {
          entity: {
            id: 2,
            name: 'about',
            alias: 'About'
          }
        },
        {
          entity: {
            id: 3,
            name: 'content',
            alias: 'Content'
          }
        }
        // {
        //   entity: {
        //     id: 4,
        //     name: 'User',
        //     alias: 'User'
        //   },
        //   childs: [
        //     {
        //       entity: {
        //         id: 41,
        //         name: 'login',
        //         alias: 'Login'
        //       }
        //     },
        //     {
        //       entity: {
        //         id: 42,
        //         name: 'register',
        //         alias: 'Register'
        //       }
        //     }
        //   ]
        // }
      ]
    }
  },
  methods: {
    handleSelect () {}
  }
}
</script>

<style>
  .el-header {
    background-color: #343A40;
    color: #CCCDCF;
  }
  .el-main {
    margin-left:15% ;
    margin-right:15% ;
  }
  .home-container {
    height: 100%;
  }
</style>
