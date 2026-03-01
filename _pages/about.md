---
permalink: /
title: "Welcome to Anbu Huang's HomePage"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<p><img src="../images/bay.png" alt="" width="100%"></p>



<p style="font-family=times"><font size=5><b>About me:</b></font></p>

I study **diffusion/flow-matching models** and **multimodal foundation models**, for which you can find more in my [blog post](https://innovation-cat.github.io/year-archive/).

- **On the Generative side**: I study the “impossible triangle” of high fidelity, fast sampling, and strong controllability—designing training and inference methods (distillation, guidance, solver-aware sampling) that expand the Pareto frontier.

- **On the Multimodal side**: I study to build native multimodal architectures that achieve a true unification of understanding and generation, enabling AI to perceive, reason, and create within a single, cohesive framework.

**Contact Me:** <strong style="color: #1D4ED8; font-weight: bold; text-decoration: underline;">huanganbu@gmail.com</strong>

---

**Previously**: I have also worked on areas including recommender systems, federated learning and AI safety. I have published multiple research papers at AI conferences such as ICLR and AAAI ([Publication](https://innovation-cat.github.io/publications/)). 


## Selected Publications



<style>
/* ===== Publications: 1/3 image + 2/3 text ===== */
.pub-list {
  margin-top: 1.2rem;
}

.pub-item {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); /* 左1/3，右2/3 */
  gap: 1.5rem;
  margin: 2rem 0 2.6rem 0;
  align-items: stretch; /* 左右上下对齐（同高） */
}

.pub-left,
.pub-right {
  min-width: 0;
}

/* 左侧图片区域 */
.pub-left {
  display: flex;
  align-items: stretch;
}

.pub-left img {
  width: 100%;
  height: 100%;              /* 跟右侧同高 */
  object-fit: contain;       /* 不裁剪，完整显示图（如果想铺满可改 cover） */
  display: block;
  border: 3px solid #c8d4ff;
  border-radius: 4px;
  background: #fff;
}

/* 右侧文字区域 */
.pub-right {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}

.pub-title {
  margin: 0 0 0.35rem 0;
  font-size: 1.1rem;
  line-height: 1.15;
  font-weight: 500;
  color: #111;
}

.pub-title a {
  color: inherit;
  text-decoration: none;
}

.pub-title a:hover {
  text-decoration: underline;
}

.pub-authors {
  font-size: 1rem;
  line-height: 1.35;
  color: #222;
  margin-bottom: 0.25rem;
}

.pub-authors a {
  color: inherit;
  text-decoration: underline;
  text-underline-offset: 3px;
}

.pub-venue {
  font-size: 0.9rem;
  line-height: 1.35;
  color: #4b5563;
  font-style: italic;
  margin-bottom: 0.3rem;
}

.pub-highlight {
  font-size: 1.05rem;
  font-weight: 500;
  color: #ff6a00;
  margin-bottom: 0.55rem;
}

.pub-btns {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.pub-btn {
  display: inline-block;
  padding: 0.15rem 0.28rem;
  border: 2px solid #222;
  border-radius: 2px;
  color: #111;
  text-decoration: none;
  font-size: 0.8rem;
  font-weight: 500;
  line-height: 1.2;
}

.pub-btn:hover {
  background: #f5f5f5;
  text-decoration: none;
}

/* 移动端：上下布局 */
@media (max-width: 900px) {
  .pub-item {
    grid-template-columns: 1fr;
    gap: 0.9rem;
    align-items: start;
  }

  .pub-left img {
    height: auto; /* 手机上恢复自然高度 */
  }

  .pub-title {
    font-size: 1.25rem;
  }

  .pub-authors,
  .pub-venue,
  .pub-highlight {
    font-size: 0.98rem;
  }
}
</style>


<div class="pub-list">

  <!-- ===== Publication 1 ===== -->
  <div class="pub-item">
    <div class="pub-left">
      <img src="../assets/images/about/p1.png" alt="Paper thumbnail">
    </div>

    <div class="pub-right">
      <h3 class="pub-title">
        <a href="https://iclr-blogposts.github.io/2026/blog/2026/flow-map-learning/">
          <strong>From Trajectories to Operators — A Unified Flow Map Perspective on Generative Modeling</strong>
        </a>
      </h3>

      <div class="pub-authors">
        Anbu Huang
      </div>

      <div class="pub-venue">
        <strong>ICLR 2026 BlogPost Track.</strong>
      </div>

      <!-- 可选：摘要（点击 ABSTRACT 跳转） -->
      <div id="abs-ctcm-2025" style="margin-bottom:0.5rem; font-size:0.95rem; color:#333;">
        <strong>Abstract.</strong>  we reframe continuous-time generative modeling from integrating trajectories to learning two-time operators (flow maps). This operator view unifies diffusion, flow matching, and consistency model. 
      </div>
      
      <div class="pub-btns">
        <a class="pub-btn" href="https://iclr-blogposts.github.io/2026/blog/2026/flow-map-learning/">Paper</a>
      </div>

      
      
    </div>
  </div>


  <!-- ===== Publication 2 ===== -->
  <div class="pub-item">
    <div class="pub-left">
      <img src="../assets/images/about/p2.jpg" alt="Paper thumbnail">
    </div>

    <div class="pub-right">
      <h3 class="pub-title">
        <a href="https://iclr-blogposts.github.io/2026/blog/2026/diffusion-inverse-problems/">
         <strong>Navigating the Manifold —  A Geometric Perspective on Diffusion-Based Inverse Problems</strong>
        </a>
      </h3>

      <div class="pub-authors">
        Anbu Huang
      </div>

      <div class="pub-venue">
        <strong>ICLR 2026 BlogPost Track.</strong>
      </div>

      <!-- 可选：摘要（点击 ABSTRACT 跳转） -->
      <div id="abs-ctcm-2025" style="margin-bottom:0.5rem; font-size:0.95rem; color:#333;">
        <strong>Abstract.</strong>   We show that a wide range of methods mostly instantiate two operator-splitting paradigms, i.e., posterior-guided sampling and clean-space local-MAP optimization.  
      </div>
      
      <div class="pub-btns">
        <a class="pub-btn" href="https://iclr-blogposts.github.io/2026/blog/2026/diffusion-inverse-problems/">Paper</a>
      </div>

      
      
    </div>
  </div>

</div>


