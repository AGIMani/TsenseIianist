import unittest
import mujoco
from tsensei_v3_2_pianist.preview_camera import configure_camera


class ReviewCameraTest(unittest.TestCase):
    def test_actual_gl_camera_projection(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1" type="sphere"/></worldbody></mujoco>')
        data=mujoco.MjData(model);scene=mujoco.MjvScene(model,maxgeom=10);camera=mujoco.MjvCamera()
        for view in ('front','performer','overview'):
            config=configure_camera(camera,model,view=view)
            mujoco.mjv_updateScene(model,data,mujoco.MjvOption(),None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            self.assertEqual(bool(scene.camera[0].orthographic),view!='overview')
            if view!='overview':
                self.assertAlmostEqual(scene.camera[0].frustum_top-scene.camera[0].frustum_bottom,config['fovy'],places=6)


if __name__=='__main__':unittest.main()
