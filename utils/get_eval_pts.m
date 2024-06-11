## Author: Clara Rodrigo Gonzalez

function pts = get_eval_pts(frame, view, savedir, outdir)

    savedir = '/media/clararg/8TB HDD/Data/STRAUS/Clara/new/';

    addpath(genpath('/home/clararg/Documents/Scripts/MoCo/moco_loco/Heart_dataset/tissue'))
    mm = 0.001;
    
    mesh = load([savedir,'/int_mesh_2490f.mat']).m;

    [x,y,z,~] = gen_heart_scat_v5(mesh, frame, 1, 0, savedir);
    [x,y,z,~] = move_view([x,y,z], zeros([length(x),1]), frame, 'PLA', 1);
    dx = zeros([length(x),1]);
    dy = dx;
    dz = dx;

    pts = [x,y,z,dx,dy,dz];
    
    save(strcat(outdir,'eval_pts_',num2str(frame),'.mat'),"pts","frame")
end